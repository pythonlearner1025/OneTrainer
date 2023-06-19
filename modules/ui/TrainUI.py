import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk
import torch
from PIL.Image import Image

from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.ConceptTab import ConceptTab
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.DataType import DataType
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossFunction import LossFunction
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class TrainUI(ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    status_label: ctk.CTkLabel | None
    training_button: ctk.CTkButton | None
    training_callbacks: TrainCallbacks | None
    training_commands: TrainCommands | None

    def __init__(self):
        super(TrainUI, self).__init__()

        self.title("OneTrainer")
        self.geometry("1100x700")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.train_args = TrainArgs.default_values()
        self.ui_state = UIState(self, self.train_args)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = None
        self.training_button = None
        self.tabview = None

        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

    def close(self):
        self.top_bar_component.save_default()

    def top_bar(self, master):
        return TopBar(master, self.train_args, self.ui_state, self.change_training_method)

    def bottom_bar(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=2, column=0, sticky="nsew")

        self.set_step_progress, self.set_epoch_progress = components.double_progress(frame, 0, 0, "step", "epoch")

        self.status_label = components.label(frame, 0, 1, "")

        # padding
        frame.grid_columnconfigure(2, weight=1)

        # tensorboard button
        components.button(frame, 0, 3, "Tensorboard", self.open_tensorboard)

        # training button
        self.training_button = components.button(frame, 0, 4, "Start Training", self.start_training)

        # export button
        self.export_button = components.button(frame, 0, 5, "Export", self.export_training,
                                               tooltip="Export the current configuration as a script to run without a UI")

        return frame

    def content_frame(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(frame)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab(self.tabview.add("general"))
        self.model_tab(self.tabview.add("model"))
        self.data_tab(self.tabview.add("data"))
        self.concepts_tab(self.tabview.add("concepts"))
        self.training_tab(self.tabview.add("training"))
        self.sampling_tab(self.tabview.add("sampling"))
        self.backup_tab(self.tabview.add("backup"))
        self.tools_tab(self.tabview.add("tools"))

        self.change_training_method(self.train_args.training_method)

        return frame

    def general_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=0)
        master.grid_columnconfigure(3, weight=1)

        # workspace dir
        components.label(master, 0, 0, "Workspace Directory",
                         tooltip="The directory where all files of this training run is saved")
        components.dir_entry(master, 0, 1, self.ui_state, "workspace_dir")

        # cache dir
        components.label(master, 1, 0, "Cache Directory",
                         tooltip="The directory where cached data is saved")
        components.dir_entry(master, 1, 1, self.ui_state, "cache_dir")

        # debug
        components.label(master, 2, 0, "Only Cache",
                         tooltip="Only populate the cache, without any training")
        components.switch(master, 2, 1, self.ui_state, "only_cache")

        # debug
        components.label(master, 3, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")
        components.switch(master, 3, 1, self.ui_state, "debug_mode")

        components.label(master, 4, 0, "Debug Directory",
                         tooltip="The directory where debug data is saved")
        components.file_entry(master, 4, 1, self.ui_state, "debug_dir")

        components.label(master, 5, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")
        components.switch(master, 5, 1, self.ui_state, "tensorboard")

    def model_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # base model
        components.label(master, 0, 0, "Base Model")
        components.file_entry(
            master, 0, 1, self.ui_state, "base_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # model type
        components.label(master, 0, 3, "Model Type")
        components.options_kv(master, 0, 4, [
            ("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15),
            ("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20),
            ("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21),
            ("Kandinsky 2.1", ModelType.KANDINSKY_21),
        ], self.ui_state, "model_type")

        # output model destination
        components.label(master, 1, 0, "Model Output Destination")
        components.file_entry(master, 1, 1, self.ui_state, "output_model_destination", is_output=True)

        # output format
        components.label(master, 1, 3, "Output Format")
        components.options_kv(master, 1, 4, [
            ("Diffusers", ModelFormat.DIFFUSERS),
            ("Checkpoint", ModelFormat.CKPT),
            ("Safetensors", ModelFormat.SAFETENSORS),
        ], self.ui_state, "output_model_format")

        # output data type
        components.label(master, 2, 0, "Output Data Type")
        components.options_kv(master, 2, 1, [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
        ], self.ui_state, "output_dtype")

    def data_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # circular mask generation
        components.label(master, 0, 0, "Circular Mask Generation")
        components.switch(master, 0, 1, self.ui_state, "circular_mask_generation")

        # random rotate and crop
        components.label(master, 1, 0, "Random Rotate and Crop")
        components.switch(master, 1, 1, self.ui_state, "random_rotate_and_crop")

        # aspect ratio bucketing
        components.label(master, 2, 0, "Aspect Ratio Bucketing")
        components.switch(master, 2, 1, self.ui_state, "aspect_ratio_bucketing")

        # latent caching
        components.label(master, 3, 0, "Latent Caching")
        components.switch(master, 3, 1, self.ui_state, "latent_caching")

        # latent caching epochs
        components.label(master, 4, 0, "Latent Caching Epochs")
        components.entry(master, 4, 1, self.ui_state, "latent_caching_epochs")

    def concepts_tab(self, master):
        ConceptTab(master, self.train_args, self.ui_state)

    def training_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)
        master.grid_columnconfigure(5, minsize=50)
        master.grid_columnconfigure(6, weight=0)
        master.grid_columnconfigure(7, weight=1)

        # column 1
        # optimizer
        components.label(master, 0, 0, "Optimizer")
        components.options(master, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer")

        # learning rate scheduler
        components.label(master, 1, 0, "Learning Rate Scheduler")
        components.options(master, 1, 1, [str(x) for x in list(LearningRateScheduler)], self.ui_state,
                           "learning_rate_scheduler")

        # learning rate
        components.label(master, 2, 0, "Learning Rate")
        components.entry(master, 2, 1, self.ui_state, "learning_rate")

        # learning rate warmup steps
        components.label(master, 3, 0, "Learning Rate Warmup Steps")
        components.entry(master, 3, 1, self.ui_state, "learning_rate_warmup_steps")

        # learning rate cycles
        components.label(master, 4, 0, "Learning Rate Cycles")
        components.entry(master, 4, 1, self.ui_state, "learning_rate_cycles")

        # weight decay
        components.label(master, 5, 0, "Weight Decay")
        components.entry(master, 5, 1, self.ui_state, "weight_decay")

        # loss function
        components.label(master, 6, 0, "Loss Function")
        components.options(master, 6, 1, [str(x) for x in list(LossFunction)], self.ui_state, "loss_function")

        # epochs
        components.label(master, 7, 0, "Epochs")
        components.entry(master, 7, 1, self.ui_state, "epochs")

        # batch size
        components.label(master, 8, 0, "Batch Size")
        components.entry(master, 8, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(master, 9, 0, "Accumulation Steps")
        components.entry(master, 9, 1, self.ui_state, "gradient_accumulation_steps")

        # attention mechanism
        components.label(master, 10, 0, "Attention")
        components.options(master, 10, 1, [str(x) for x in list(AttentionMechanism)], self.ui_state,
                           "attention_mechanism")

        # column 2
        # train text encoder
        components.label(master, 0, 3, "Train Text Encoder")
        components.switch(master, 0, 4, self.ui_state, "train_text_encoder")

        # train text encoder epochs
        components.label(master, 1, 3, "Train Text Encoder Epochs")
        components.entry(master, 1, 4, self.ui_state, "train_text_encoder_epochs")

        # text encoder learning rate
        components.label(master, 2, 3, "Text Encoder Learning Rate")
        components.entry(master, 2, 4, self.ui_state, "text_encoder_learning_rate")

        # text encoder layer skip (clip skip)
        components.label(master, 3, 3, "Clip Skip")
        components.entry(master, 3, 4, self.ui_state, "text_encoder_layer_skip")

        # offset noise weight
        components.label(master, 5, 3, "Offset Noise Weight")
        components.entry(master, 5, 4, self.ui_state, "offset_noise_weight")

        # rescale noise scheduler to zero terminal SNR
        components.label(master, 6, 3, "Rescale Noise Scheduler")
        components.switch(master, 6, 4, self.ui_state, "rescale_noise_scheduler_to_zero_terminal_snr")

        # train dtype
        components.label(master, 7, 3, "Train Data Type")
        components.options_kv(master, 7, 4, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], self.ui_state, "train_dtype")

        # resolution
        components.label(master, 8, 3, "Resolution")
        components.entry(master, 8, 4, self.ui_state, "resolution")

        # column 3
        # train unet
        components.label(master, 0, 6, "Train UNet")
        components.switch(master, 0, 7, self.ui_state, "train_unet")

        # train unet epochs
        components.label(master, 1, 6, "Train UNet Epochs")
        components.entry(master, 1, 7, self.ui_state, "train_unet_epochs")

        # unet learning rate
        components.label(master, 2, 6, "Unet Learning Rate")
        components.entry(master, 2, 7, self.ui_state, "unet_learning_rate")

        # Masked Training
        components.label(master, 5, 6, "Masked Training")
        components.switch(master, 5, 7, self.ui_state, "masked_training")

        # unmasked probability
        components.label(master, 6, 6, "Unmasked Probability")
        components.entry(master, 6, 7, self.ui_state, "unmasked_probability")

        # unmasked weight
        components.label(master, 7, 6, "Unmasked Weight")
        components.entry(master, 7, 7, self.ui_state, "unmasked_weight")

        # normalize masked area loss
        components.label(master, 8, 6, "Normalize Masked Area Loss")
        components.switch(master, 8, 7, self.ui_state, "normalize_masked_area_loss")

        # max noising strength
        components.label(master, 9, 6, "Max Noising Strength")
        components.entry(master, 9, 7, self.ui_state, "max_noising_strength")

    def sampling_tab(self, master):
        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # sample after
        top_frame = ctk.CTkFrame(master=master, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="nsew")

        components.label(top_frame, 0, 0, "Sample After")
        components.time_entry(top_frame, 0, 1, self.ui_state, "sample_after", "sample_after_unit")

        # table
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        SamplingTab(frame, self.train_args, self.ui_state)

    def backup_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # sample after
        components.label(master, 0, 0, "Backup After")
        components.time_entry(master, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

        # optimizer
        components.label(master, 1, 0, "Backup Before Save")
        components.switch(master, 1, 1, self.ui_state, "backup_before_save")

    def lora_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # extra model
        components.label(master, 0, 0, "LoRA base model")
        components.file_entry(
            master, 0, 1, self.ui_state, "extra_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # lora rank
        components.label(master, 1, 0, "LoRA rank")
        components.entry(master, 1, 1, self.ui_state, "lora_rank")

        # lora rank
        components.label(master, 2, 0, "LoRA alpha")
        components.entry(master, 2, 1, self.ui_state, "lora_alpha")

        return master

    def embedding_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # extra model
        components.label(master, 0, 0, "Base embedding")
        components.file_entry(
            master, 0, 1, self.ui_state, "extra_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # token count
        components.label(master, 1, 0, "Token count")
        components.entry(master, 1, 1, self.ui_state, "token_count")

        # initial embedding text
        components.label(master, 2, 0, "Initial embedding text")
        components.entry(master, 2, 1, self.ui_state, "initial_embedding_text")

        return master

    def tools_tab(self, master):
        pass

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if training_method != TrainingMethod.LORA and "LoRA" in self.tabview._tab_dict:
            self.tabview.delete("LoRA")
        if training_method != TrainingMethod.EMBEDDING and "embedding" in self.tabview._tab_dict:
            self.tabview.delete("embedding")

        if training_method == TrainingMethod.LORA and "LoRA" not in self.tabview._tab_dict:
            self.lora_tab(self.tabview.add("LoRA"))
        if training_method == TrainingMethod.EMBEDDING and "embedding" not in self.tabview._tab_dict:
            self.embedding_tab(self.tabview.add("embedding"))

    def open_tensorboard(self):
        webbrowser.open("http://localhost:6006/", new=0, autoraise=False)

    def on_update_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_sample)
        self.set_epoch_progress(train_progress.epoch, max_epoch)
        pass

    def on_update_status(self, status: str):
        self.status_label.configure(text=status)
        pass

    def on_sample(self, sample: Image):
        pass

    def training_thread_function(self):
        error_caught = False

        callbacks = TrainCallbacks(
            on_update_progress=self.on_update_progress,
            on_update_status=self.on_update_status,
            on_sample=self.on_sample,
        )

        trainer = GenericTrainer(self.train_args, callbacks, self.training_commands)

        try:
            trainer.start()
            trainer.train()
        except:
            error_caught = True
            traceback.print_exc()

        if self.train_args.backup_before_save:
            trainer.end()

        torch.cuda.empty_cache()

        if error_caught:
            self.on_update_status("error: check the console for more information")
        else:
            self.on_update_status("stopped")

        self.training_thread = None
        self.training_button.configure(text="Start Training", state="normal")

    def start_training(self):
        if self.training_thread is None:
            self.top_bar_component.save_default()

            self.training_button.configure(text="Stop Training", state="normal")

            self.training_commands = TrainCommands()

            self.training_thread = threading.Thread(target=self.training_thread_function)
            self.training_thread.start()
        else:
            self.training_button.configure(state="disabled")
            self.on_update_status("stopping")
            self.training_commands.stop()

    def export_training(self):
        args = self.train_args.to_args()
        command = "python scripts/train.py " + args

        file_path = filedialog.asksaveasfilename(filetypes=[
            ("All Files", "*.*"),
            ("Batch", "*.bat"),
            ("Shell", "*.sh"),
        ], initialdir=".", initialfile="train.bat")

        if file_path:
            with open(file_path, "w") as f:
                f.write(command)
