import customtkinter as ctk
from modules.util.ui import components
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.Optimizer import Optimizer
import math

class OptimizerParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.train_args = TrainArgs.default_values()
        self.ui_state = ui_state

        self.title("Optimizer Settings")
        self.geometry("800x400")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.frame = ctk.CTkFrame(self)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, minsize=50)
        self.frame.grid_columnconfigure(3, weight=0)
        self.frame.grid_columnconfigure(4, weight=1)
        
        components.button(self, 1, 0, "ok", self.__ok)
        self.button = None
        self.main_frame(self.frame)   
    
    def __ok(self):
        self.destroy()
        
    def create_dynamic_ui(self, selected_optimizer, master, components, ui_state, defaults=False):
        # Lookup for keys that belong to each optimizer.
        OPTIMIZER_KEY_MAP = {
            'RMSPROP_8BIT': ['weight_decay', 'alpha', 'momentum', 'centered', 'min_8bit_size', 'percentile_clipping', 'block_wise'],
            'SGD': ['weight_decay', 'foreach'],
            'SGD_8BIT': ['momentum', 'dampening', 'weight_decay', 'nesterov'],
            'ADAM': ['weight_decay', 'eps', 'foreach', 'fused'],
            'ADAMW': ['weight_decay', 'eps', 'foreach', 'fused'],
            'ADAM_8BIT': ['weight_decay', 'eps', 'min_8bit_size', 'percentile_clipping', 'block_wise', 'is_paged'],
            'ADAMW_8BIT': ['weight_decay', 'eps', 'min_8bit_size', 'percentile_clipping', 'block_wise', 'is_paged'],
            'ADAGRAD': ['weight_decay', 'eps', 'lr_decay', 'initial_accumulator_value'],
            'ADAGRAD_8BIT': ['weight_decay', 'eps', 'lr_decay', 'initial_accumulator_value', 'min_8bit_size', 'percentile_clipping', 'block_wise'],
            'RMSPROP': ['weight_decay', 'eps', 'alpha', 'momentum', 'centered'],
            'RMSPROP_8BIT': ['weight_decay', 'eps', 'alpha', 'momentum', 'centered', 'min_8bit_size', 'percentile_clipping', 'block_wise'],
            'LION': ['weight_decay'],
            'LARS': ['weight_decay', 'momentum', 'dampening', 'nesterov', 'max_unorm'],
            'LARS_8BIT': ['weight_decay', 'momentum', 'dampening', 'nesterov', 'min_8bit_size', 'percentile_clipping', 'max_unorm'],
            'LAMB': ['weight_decay', 'betas', 'bias_correction', 'amsgrad', 'adam_w_mode', 'percentile_clipping', 'block_wise', 'max_unorm'],
            'LAMB_8BIT': ['weight_decay', 'betas', 'bias_correction', 'amsgrad', 'adam_w_mode', 'min_8bit_size', 'percentile_clipping', 'block_wise', 'max_unorm'],
            'LION_8BIT': ['weight_decay', 'betas', 'min_8bit_size', 'percentile_clipping', 'block_wise', 'is_paged'],
            'DADAPT_SGD': ['weight_decay'],
            'DADAPT_ADAM': ['weight_decay'],
            'DADAPT_ADAN': ['weight_decay'],
            'DADAPT_ADA_GRAD': ['weight_decay'],
            'DADAPT_LION': ['weight_decay'],
            'PRODIGY': ['betas', 'beta3', 'eps', 'weight_decay', 'decouple', 'use_bias_correction', 'safeguard_warmup', 'd0', 'd_coef', 'growth_rate', 'fsdp_in_use'],
            'ADAFACTOR': ['eps_tuple', 'clip_threshold', 'decay_rate', 'beta1', 'weight_decay', 'scale_parameter', 'relative_step', 'warmup_init']
        }

        #Lookup for the title and tooltip for a key
        KEY_DETAIL_MAP = {
            'adam_w_mode': {'title': 'Adam W Mode', 'tooltip': 'Whether to use weight decay correction for Adam optimizer.', 'type': 'bool'},
            'alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'amsgrad': {'title': 'AMSGrad', 'tooltip': 'Whether to use the AMSGrad variant for Adam.', 'type': 'bool'},
            'beta1': {'title': 'Beta1', 'tooltip': 'Momentum term.', 'type': 'float'},
            'beta3': {'title': 'Beta3', 'tooltip': 'Coefficient for computing the Prodigy stepsize.', 'type': 'float'},
            'betas': {'title': 'Betas', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'tuple[float, float]'},
            'bias_correction': {'title': 'Bias Correction', 'tooltip': 'Whether to use bias correction in optimization algorithms like Adam.', 'type': 'bool'},
            'block_wise': {'title': 'Block Wise', 'tooltip': 'Whether to perform block-wise model update.', 'type': 'bool'},
            'centered': {'title': 'Centered', 'tooltip': 'Whether to center the gradient before scaling. Great for stabilizing the training process.', 'type': 'bool'},
            'clip_threshold': {'title': 'Clip Threshold', 'tooltip': 'Clipping value for gradients.', 'type': 'float'},
            'd0': {'title': 'Initial D', 'tooltip': 'Initial D estimate for D-adaptation.', 'type': 'float'},
            'd_coef': {'title': 'D Coefficient', 'tooltip': 'Coefficient in the expression for the estimate of d.', 'type': 'float'},
            'dampening': {'title': 'Dampening', 'tooltip': 'Dampening for momentum.', 'type': 'float'},
            'decay_rate': {'title': 'Decay Rate', 'tooltip': 'Rate of decay for moment estimation.', 'type': 'float'},
            'decouple': {'title': 'Decouple', 'tooltip': 'Use AdamW style decoupled weight decay.', 'type': 'bool'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'eps_tuple': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'tuple[float, float]'},
            'foreach': {'title': 'ForEach', 'tooltip': 'If true, apply the optimizer to each parameter independently.', 'type': 'bool'},
            'fsdp_in_use': {'title': 'FSDP in Use', 'tooltip': 'Flag for using sharded parameters.', 'type': 'bool'},
            'fused': {'title': 'Fused', 'tooltip': 'Whether to use a fused implementation if available.', 'type': 'bool'},
            'growth_rate': {'title': 'Growth Rate', 'tooltip': 'Limit for D estimate growth rate.', 'type': 'float'},
            'initial_accumulator_value': {'title': 'Initial Accumulator Value', 'tooltip': 'Initial value for Adagrad optimizer.', 'type': 'float'},
            'is_paged': {'title': 'Is Paged', 'tooltip': 'Whether the optimizer\'s internal state should be paged to CPU.', 'type': 'bool'},
            'lr_decay': {'title': 'LR Decay', 'tooltip': 'Rate at which learning rate decreases.', 'type': 'float'},
            'max_unorm': {'title': 'Max Unorm', 'tooltip': 'Maximum value for gradient clipping by norms.', 'type': 'float'},
            'min_8bit_size': {'title': 'Min 8bit Size', 'tooltip': 'Minimum tensor size for 8-bit quantization.', 'type': 'int'},
            'momentum': {'title': 'Momentum', 'tooltip': 'Factor to accelerate SGD in relevant direction.', 'type': 'float'},
            'nesterov': {'title': 'Nesterov', 'tooltip': 'Whether to enable Nesterov momentum.', 'type': 'bool'},
            'percentile_clipping': {'title': 'Percentile Clipping', 'tooltip': 'Gradient clipping based on percentile values.', 'type': 'float'},
            'relative_step': {'title': 'Relative Step', 'tooltip': 'Whether to use a relative step size.', 'type': 'bool'},
            'safeguard_warmup': {'title': 'Safeguard Warmup', 'tooltip': 'Avoid issues during warm-up stage.', 'type': 'bool'},
            'scale_parameter': {'title': 'Scale Parameter', 'tooltip': 'Whether to scale the parameter or not.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'warmup_init': {'title': 'Warmup Initialization', 'tooltip': 'Whether to warm-up the optimizer initialization.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
        }
        
        optimizers_defaults = {
            "ADAFACTOR": {
                "eps_tuple": (1e-30, 1e-3),
                "clip_threshold": 1.0,
                "decay_rate": -0.8,
                "beta1": None,
                "weight_decay": 0.0,
                "scale_parameter": True,
                "relative_step": True,
                "warmup_init": False,
            },
            "ADAGRAD": {
                "lr_decay": 0,
                "weight_decay": 0,
                "initial_accumulator_value": 0,
                "eps": 1e-10,
                "optim_bits": 32,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
            },
            "ADAGRAD_8BIT": {
                "lr_decay": 0,
                "weight_decay": 0,
                "initial_accumulator_value": 0,
                "eps": 1e-10,
                "optim_bits": 8,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
            },
            "ADAM_8BIT": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "optim_bits": 32,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
                "is_paged": False,
            },
            "ADAMW_8BIT": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 1e-2,
                "amsgrad": False,
                "optim_bits": 32,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
                "is_paged": False,
            },
            "LAMB": {
                "bias_correction": True,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "adam_w_mode": True,
                "optim_bits": 32,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": False,
                "max_unorm": 1.0,
            },
            "LAMB_8BIT": {
                "bias_correction": True,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "adam_w_mode": True,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": False,
                "max_unorm": 1.0,
            },
            "LARS": {
                "momentum": 0,
                "dampening": 0,
                "weight_decay": 0,
                "nesterov": False,
                "optim_bits": 32,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "max_unorm": 0.02,
            },
            "LARS_8BIT": {
                "momentum": 0,
                "dampening": 0,
                "weight_decay": 0,
                "nesterov": False,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "max_unorm": 0.02,
            },
            "LION_8BIT": {
                "betas": (0.9, 0.99),
                "weight_decay": 0,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
                "is_paged": False,
            },
            "RMSPROP": {
                "alpha": 0.99,
                "eps": 1e-8,
                "weight_decay": 0,
                "momentum": 0,
                "centered": False,
                "optim_bits": 32,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
            },
            "RMSPROP_8BIT": {
                "alpha": 0.99,
                "eps": 1e-8,
                "weight_decay": 0,
                "momentum": 0,
                "centered": False,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
            },
            "SGD_8BIT": {
                "momentum": 0,
                "dampening": 0,
                "weight_decay": 0,
                "nesterov": False,
                "min_8bit_size": 4096,
                "percentile_clipping": 100,
                "block_wise": True,
            },
            "PRODIGY": {
                "betas": (0.9, 0.999),
                "beta3": None,
                "eps": 1e-8,
                "weight_decay": 0,
                "decouple": True,
                "use_bias_correction": False, 
                "safeguard_warmup": False,
                "d0": 1e-6,
                "d_coef": 1.0,
                "growth_rate": float('inf'),
                "fsdp_in_use": False,
            },
            "DADAPT_ADA_GRAD": {
                "momentum": 0,
                "log_every": 0,
                "weight_decay": 0.0,
                "eps": 0.0,
                "d0": 1e-6,
                "growth_rate": float('inf'),
            },
            "DADAPT_ADAN": {
                "betas": (0.98, 0.92, 0.99),
                "eps": 1e-8,
                "weight_decay": 0.02,
                "no_prox": False,
                "log_every": 0,
                "d0": 1e-6,
                "growth_rate": float('inf'),
            },
            "DADAPT_ADAM": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "log_every": 0,
                "decouple": False,
                "use_bias_correction": False,
                "d0": 1e-6,
                "growth_rate": float('inf'),
                "fsdp_in_use": False,
            },
            "DADAPT_SGD": {
                "momentum": 0.0,
                "weight_decay": 0,
                "log_every": 0,
                "d0": 1e-6,
                "growth_rate": float('inf'),
                "fsdp_in_use": False,
            },
            "DADAPT_LION": {
                "betas": (0.9, 0.99),
                "weight_decay": 0.0,
                "log_every": 0,
                "d0": 1e-6,
                "fsdp_in_use": False,
            },
            "ADAM": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "foreach": None,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
                "fused": None
            },
            "ADAMW": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 1e-2,
                "amsgrad": False,
                "foreach": None,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
                "fused": None
            },
            "SGD": {
                "momentum": 0,
                "dampening": 0,
                "weight_decay": 0,
                "nesterov": False,
                "foreach": None,
                "maximize": False,
                "differentiable": False
            }
     
        }

        
        if not self.winfo_exists():  # check if this window isn't open
            return
            
        for idx, key in enumerate(OPTIMIZER_KEY_MAP[selected_optimizer]):
            arg_info = KEY_DETAIL_MAP[key]
            
            title = arg_info['title']
            tooltip = arg_info['tooltip']
            type = arg_info['type']
            
            row = math.floor(idx / 2) + 1
            col = 3 * (idx % 2) 
            
            components.label(master, row, col, title, tooltip=tooltip)
            override_value = None
            
            if defaults and key in optimizers_defaults[selected_optimizer]:
                override_value = optimizers_defaults[selected_optimizer][key]

            if type != 'bool':
                components.entry(master, row, col+1, ui_state, key, override_value=override_value)
            else:
                components.switch(master, row, col+1, ui_state, key, override_value=override_value)
        
    def main_frame(self, master):
   
        # Optimizer
        components.label(master, 0, 0, "Optimizer", tooltip="The type of optimizer")
        components.options(master, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer")
        
        # Defaults Button
        components.label(master, 0, 0, "Optimizer Defaults", tooltip="Load default settings for the selected optimizer")
        components.button(self.frame, 0, 4, "Load Defaults", self.load_defaults, tooltip="Load default settings for the selected optimizer")

        selected_optimizer = self.ui_state.vars['optimizer'].get()
        
        self.ui_state.vars['optimizer'].trace_add('write', self.on_optimizer_change)
        self.create_dynamic_ui(selected_optimizer, master, components, self.ui_state)
        
    def on_optimizer_change(self, *args):
        if not self.winfo_exists():  # check if this window isn't open
            return
        selected_optimizer = self.ui_state.vars['optimizer'].get()
        self.clear_dynamic_ui(self.frame)
        self.create_dynamic_ui(selected_optimizer, self.frame, components, self.ui_state)
        
    def load_defaults(self):
        if not self.winfo_exists():  # check if this window isn't open
            return
        selected_optimizer = self.ui_state.vars['optimizer'].get()
        self.clear_dynamic_ui(self.frame)
        self.create_dynamic_ui(selected_optimizer, self.frame, components, self.ui_state, defaults=True)

            
    def clear_dynamic_ui(self, master):
        try:
            for widget in master.winfo_children():
                grid_info = widget.grid_info()
                if int(grid_info["row"]) >= 1:
                    widget.destroy()
        except _tkinter.TclError as e:
            pass
            
