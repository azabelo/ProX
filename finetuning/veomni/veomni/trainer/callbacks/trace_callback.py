# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List

from tqdm import trange

from ...distributed.parallel_state import get_parallel_state
from ...utils import helper
from ...utils.dist_utils import all_reduce
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


class MoERouterMonitorCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.monitor = None

        args: "VeOmniArguments" = self.trainer.args
        if not args.train.wandb.enable:
            logger.info_rank0("MoE router monitor disabled (wandb not enabled).")
            return
        if args.train.moe_load_balance_monitor_interval <= 0:
            logger.info_rank0("MoE router monitor disabled (moe_load_balance_monitor_interval=0).")
            return

        config = self.trainer.model_config
        if hasattr(config, "num_experts"):
            from ...utils.moe_monitor import MoERouterMonitor, set_active_monitor

            self.monitor = MoERouterMonitor(config.num_experts)
            set_active_monitor(self.monitor)
            logger.info_rank0(
                f"MoE router monitor enabled: num_experts={config.num_experts}, "
                f"interval={args.train.moe_load_balance_monitor_interval}"
            )
        else:
            logger.warning_rank0(
                "moe_load_balance_monitor_interval > 0 but model config has no 'num_experts'. "
                "MoE router monitor not activated."
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if (
            self.monitor
            and state.global_step % args.train.moe_load_balance_monitor_interval == 0
            and args.train.global_rank == 0
        ):
            import wandb

            load_matrix = self.monitor.get_load_matrix(current_step=state.global_step)
            num_layers = load_matrix.shape[0]
            if num_layers == 0:
                logger.warning_rank0(
                    f"Step {state.global_step}: MoE router monitor has no recorded data. "
                    "Check that router forward hooks are registered (e.g. PatchQwen3MoeTopKRouter)."
                )
                return
            from ...utils.moe_monitor import MoERouterMonitor

            image = self.monitor.create_wandb_image(load_matrix)
            vio = MoERouterMonitor.compute_vio(load_matrix)
            max_vio, min_vio, avg_vio = vio["max_vio"], vio["min_vio"], vio["avg_vio"]
            metrics = {"moe/expert_load_heatmap": image}
            for i in range(num_layers):
                metrics[f"moe/max_vio/layer_{i}"] = max_vio[i].item()
                metrics[f"moe/min_vio/layer_{i}"] = min_vio[i].item()
                metrics[f"moe/avg_vio/layer_{i}"] = avg_vio[i].item()
            metrics["moe/max_vio/max"] = max_vio.max().item()
            metrics["moe/max_vio/avg"] = max_vio.mean().item()
            metrics["moe/min_vio/max"] = min_vio.max().item()
            metrics["moe/min_vio/avg"] = min_vio.mean().item()
            metrics["moe/avg_vio/max"] = avg_vio.max().item()
            metrics["moe/avg_vio/avg"] = avg_vio.mean().item()
            wandb.log(metrics, step=state.global_step)
            logger.info_rank0(
                f"Step {state.global_step}: uploaded MoE load balance heatmap "
                f"({num_layers} layers, {load_matrix.shape[1]} experts, "
                f"steps {self.monitor._last_step_range[0]}-{self.monitor._last_step_range[1]}), "
                f"max_vio: max={vio['max_vio'].max().item():.4f} avg={vio['max_vio'].mean().item():.4f}, "
                f"min_vio: max={vio['min_vio'].max().item():.4f} avg={vio['min_vio'].mean().item():.4f}, "
                f"avg_vio: max={vio['avg_vio'].max().item():.4f} avg={vio['avg_vio'].mean().item():.4f}."
            )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is not None:
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
            logger.info_rank0("MoE router monitor disabled.")


def _wandb_sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensors / numpy scalars for wandb.log; keep wandb.Image and similar."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        mod = getattr(type(v), "__module__", "")
        if mod == "wandb.sdk.data_types.image" or mod.startswith("wandb."):
            out[k] = v
            continue
        if hasattr(v, "item"):
            try:
                out[k] = float(v.item())
                continue
            except Exception:
                pass
        if np is not None and isinstance(v, np.generic):
            out[k] = float(v)
            continue
        if isinstance(v, (float, int, bool)):
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


class WandbTraceCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self._wandb_run_active = False

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self._wandb_run_active = False
        if args.train.global_rank != 0 or not args.train.wandb.enable:
            return

        wmode = os.environ.get("WANDB_MODE", "").strip().lower()
        if wmode in ("disabled", "off", "0", "false"):
            logger.info_rank0(
                "W&B skipped (WANDB_MODE=%s). Training continues without logging.",
                wmode or "unset",
            )
            return

        from dataclasses import asdict

        import wandb

        init_kw: Dict[str, Any] = {
            "project": os.environ.get("WANDB_PROJECT") or args.train.wandb.project,
            "name": os.environ.get("WANDB_NAME") or args.train.wandb.name,
            "id": args.train.wandb.id,
            "resume": "allow" if args.train.wandb.id else None,
            "config": {**asdict(args.model), **asdict(args.data), **asdict(args.train)},
        }
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            init_kw["entity"] = entity

        try:
            wandb.init(**init_kw)
        except Exception as e:
            logger.warning_rank0(
                "W&B could not start (%s: %s). Training continues without W&B. "
                "Fix: `wandb login`, or `export WANDB_API_KEY=...`, or disable with "
                "`WANDB_MODE=disabled` or `--train.wandb.enable false`.",
                type(e).__name__,
                e,
            )
            return

        self._wandb_run_active = True

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args

        if args.train.global_rank == 0 and args.train.wandb.enable and self._wandb_run_active:
            import wandb

            payload = dict(self.trainer.step_env_metrics)
            payload["train/epoch"] = float(state.epoch)
            payload["train/global_step"] = int(state.global_step)
            wandb.log(_wandb_sanitize_metrics(payload), step=int(state.global_step))

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable and self._wandb_run_active:
            import wandb

            wandb.log(
                {"train/epoch_marker": float(state.epoch)},
                step=int(state.global_step),
            )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable and self._wandb_run_active:
            import wandb

            try:
                wandb.finish()
            except Exception:
                pass
        self._wandb_run_active = False


class ProfileTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile.start_step,
                end_step=args.train.profile.end_step,
                trace_dir=args.train.profile.trace_dir,
                record_shapes=args.train.profile.record_shapes,
                profile_memory=args.train.profile.profile_memory,
                with_stack=args.train.profile.with_stack,
                with_modules=args.train.profile.with_modules,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            if state.global_step <= args.train.profile.end_step:
                self.profiler.step()

            if state.global_step == args.train.profile.end_step:
                self.profiler.stop()


class EnvironMeterCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)

        args: "VeOmniArguments" = self.trainer.args
        self.trainer.environ_meter = helper.EnvironMeter(
            config=trainer.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            gc_steps=args.train.gc_steps,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        step_env_metrics = self.trainer.environ_meter.step(delta_time, global_step=state.global_step)

        step_train_metrics = {
            "total_loss": loss,
        }
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm

        # gather training_step_info from all ranks
        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=get_parallel_state().fsdp_group)
            for k, v in step_train_metrics.items()
        }

        if self.trainer.lr_scheduler is not None:
            lr = max(self.trainer.lr_scheduler.get_last_lr())
            step_train_metrics["training/lr"] = lr

        n_micro = max(getattr(self.trainer, "last_step_num_micro_batches", 1), 1)
        if "training/foundation_loss" in step_train_metrics:
            mean_foundation = step_train_metrics["training/foundation_loss"] / n_micro
            step_train_metrics["training/mean_foundation_loss"] = mean_foundation
            # Token perplexity ≈ exp(mean CE loss) when loss is cross-entropy in nats (natural log).
            step_train_metrics["training/perplexity"] = float(math.exp(min(mean_foundation, 80.0)))

        step_env_metrics.update(step_train_metrics)

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics


class TqdmCallback(Callback):
    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.data_loader_tqdm = trange(
            args.train_steps,
            desc=f"Epoch {state.epoch + 1}/{args.train.num_train_epochs}",
            total=args.train_steps,
            initial=self.trainer.start_step,
            disable=args.train.local_rank != 0,
        )

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        self.data_loader_tqdm.close()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        postfix = ", ".join(f"{k.split('/', 1)[-1]}: {v:.2f}" for k, v in self.trainer.step_train_metrics.items())
        self.data_loader_tqdm.set_postfix_str(postfix)
        self.data_loader_tqdm.update()
