"""
Forward-Forward Algorithm GUI

A graphical interface for training and testing the Forward-Forward algorithm.
Allows interactive training, testing, and visualization of the network's learning.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
from forward_forward import FFNetwork
import threading
import queue


class FFGui:
    """GUI for Forward-Forward Algorithm"""

    def __init__(self, root):
        self.root = root
        self.root.title("Forward-Forward Algorithm Trainer")
        self.root.geometry("900x700")

        # Network and training state
        self.network = None
        self.is_training = False
        self.training_thread = None
        self.message_queue = queue.Queue()

        # Training data
        self.X_pos = None
        self.X_neg = None

        # Setup UI
        self.setup_ui()

        # Check message queue periodically
        self.root.after(100, self.check_queue)

    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title = tk.Label(
            self.root,
            text="Forward-Forward Algorithm Trainer",
            font=("Arial", 16, "bold"),
            pady=10
        )
        title.pack()

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Configuration
        left_panel = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Network architecture
        ttk.Label(left_panel, text="Network Architecture:").grid(row=0, column=0, sticky="w", pady=5)
        self.arch_entry = ttk.Entry(left_panel, width=20)
        self.arch_entry.insert(0, "3,8,4")
        self.arch_entry.grid(row=0, column=1, pady=5)
        ttk.Label(left_panel, text="(comma-separated)").grid(row=0, column=2, sticky="w", padx=5)

        # Learning rate
        ttk.Label(left_panel, text="Learning Rate:").grid(row=1, column=0, sticky="w", pady=5)
        self.lr_entry = ttk.Entry(left_panel, width=20)
        self.lr_entry.insert(0, "0.03")
        self.lr_entry.grid(row=1, column=1, pady=5)

        # Threshold
        ttk.Label(left_panel, text="Threshold:").grid(row=2, column=0, sticky="w", pady=5)
        self.threshold_entry = ttk.Entry(left_panel, width=20)
        self.threshold_entry.insert(0, "1.0")
        self.threshold_entry.grid(row=2, column=1, pady=5)

        # Epochs
        ttk.Label(left_panel, text="Epochs:").grid(row=3, column=0, sticky="w", pady=5)
        self.epochs_entry = ttk.Entry(left_panel, width=20)
        self.epochs_entry.insert(0, "500")
        self.epochs_entry.grid(row=3, column=1, pady=5)

        # Data selection
        ttk.Label(left_panel, text="Dataset:").grid(row=4, column=0, sticky="w", pady=5)
        self.dataset_var = tk.StringVar(value="XOR")
        dataset_combo = ttk.Combobox(
            left_panel,
            textvariable=self.dataset_var,
            values=["XOR", "Random"],
            state="readonly",
            width=18
        )
        dataset_combo.grid(row=4, column=1, pady=5)

        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)

        self.init_button = ttk.Button(
            button_frame,
            text="Initialize Network",
            command=self.initialize_network
        )
        self.init_button.grid(row=0, column=0, padx=5)

        self.train_button = ttk.Button(
            button_frame,
            text="Start Training",
            command=self.start_training,
            state="disabled"
        )
        self.train_button.grid(row=0, column=1, padx=5)

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Training",
            command=self.stop_training,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=2, padx=5)

        # Test section
        test_frame = ttk.LabelFrame(left_panel, text="Test Input", padding="10")
        test_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(test_frame, text="Input (comma-separated):").grid(row=0, column=0, sticky="w", pady=5)
        self.test_input_entry = ttk.Entry(test_frame, width=30)
        self.test_input_entry.insert(0, "0,0,1")
        self.test_input_entry.grid(row=1, column=0, pady=5)

        self.test_button = ttk.Button(
            test_frame,
            text="Test Input",
            command=self.test_input,
            state="disabled"
        )
        self.test_button.grid(row=2, column=0, pady=5)

        self.test_result_label = ttk.Label(test_frame, text="", foreground="blue")
        self.test_result_label.grid(row=3, column=0, pady=5)

        # Right panel - Output
        right_panel = ttk.LabelFrame(main_frame, text="Training Output", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            right_panel,
            width=50,
            height=30,
            font=("Courier", 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            right_panel,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Status label
        self.status_label = ttk.Label(right_panel, text="Ready", foreground="green")
        self.status_label.pack(pady=5)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def log_message(self, message):
        """Add message to output text area"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def initialize_network(self):
        """Initialize the neural network"""
        try:
            # Parse architecture
            arch_str = self.arch_entry.get()
            layer_dims = [int(x.strip()) for x in arch_str.split(",")]

            if len(layer_dims) < 2:
                raise ValueError("Need at least 2 layers (input and output)")

            # Parse hyperparameters
            learning_rate = float(self.lr_entry.get())
            threshold = float(self.threshold_entry.get())

            # Create network
            self.network = FFNetwork(
                layer_dims=layer_dims,
                learning_rate=learning_rate,
                threshold=threshold
            )

            # Generate training data
            self.generate_training_data(layer_dims[0])

            # Update UI
            self.log_message("=" * 60)
            self.log_message("Network Initialized Successfully!")
            self.log_message(f"Architecture: {' -> '.join(map(str, layer_dims))}")
            self.log_message(f"Learning Rate: {learning_rate}")
            self.log_message(f"Threshold: {threshold}")
            self.log_message(f"Dataset: {self.dataset_var.get()}")
            self.log_message(f"Positive samples: {len(self.X_pos)}")
            self.log_message(f"Negative samples: {len(self.X_neg)}")
            self.log_message("=" * 60)

            self.train_button.config(state="normal")
            self.test_button.config(state="normal")
            self.status_label.config(text="Network initialized - Ready to train", foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize network:\n{str(e)}")
            self.status_label.config(text="Initialization failed", foreground="red")

    def generate_training_data(self, input_dim):
        """Generate training data based on selected dataset"""
        dataset = self.dataset_var.get()

        if dataset == "XOR":
            # XOR-like problem: last bit determines positive/negative
            self.X_pos = np.array([
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ], dtype=float)

            self.X_neg = np.array([
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]
            ], dtype=float)

        elif dataset == "Random":
            # Random data
            num_samples = 100
            self.X_pos = np.random.random((num_samples, input_dim)) * 2 - 1
            self.X_neg = np.random.random((num_samples, input_dim)) * 2 - 1

    def start_training(self):
        """Start training in a separate thread"""
        if self.is_training:
            return

        self.is_training = True
        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.init_button.config(state="disabled")
        self.status_label.config(text="Training...", foreground="orange")

        # Start training thread
        self.training_thread = threading.Thread(target=self.train_network)
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.status_label.config(text="Stopping...", foreground="orange")

    def train_network(self):
        """Training loop (runs in separate thread)"""
        try:
            epochs = int(self.epochs_entry.get())

            self.message_queue.put(("log", "\nStarting training...\n"))

            for epoch in range(epochs):
                if not self.is_training:
                    self.message_queue.put(("log", "\nTraining stopped by user.\n"))
                    break

                # Training step
                metrics = self.network.train_step(self.X_pos, self.X_neg)

                # Update progress
                progress = ((epoch + 1) / epochs) * 100
                self.message_queue.put(("progress", progress))

                # Log every 10% or at key points
                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                    msg = f"Epoch {epoch + 1}/{epochs}:\n"
                    for layer_idx in range(len(self.network.layers)):
                        pos_g = metrics['pos_goodness'][layer_idx]
                        neg_g = metrics['neg_goodness'][layer_idx]
                        msg += f"  Layer {layer_idx + 1}: Pos={pos_g:.4f}, Neg={neg_g:.4f}\n"
                    self.message_queue.put(("log", msg))

            if self.is_training:
                self.message_queue.put(("log", "\n✓ Training completed successfully!\n"))
                self.message_queue.put(("status", ("Training completed", "green")))
            else:
                self.message_queue.put(("status", ("Training stopped", "blue")))

        except Exception as e:
            self.message_queue.put(("log", f"\n✗ Error during training: {str(e)}\n"))
            self.message_queue.put(("status", ("Training failed", "red")))

        finally:
            self.message_queue.put(("training_done", None))

    def test_input(self):
        """Test a custom input"""
        if self.network is None:
            messagebox.showerror("Error", "Please initialize the network first")
            return

        try:
            # Parse input
            input_str = self.test_input_entry.get()
            test_input = np.array([[float(x.strip()) for x in input_str.split(",")]])

            # Check dimension
            expected_dim = self.network.layers[0].W.shape[0]
            if test_input.shape[1] != expected_dim:
                raise ValueError(f"Expected {expected_dim} values, got {test_input.shape[1]}")

            # Make prediction
            probabilities, goodness_values = self.network.predict(test_input)

            # Display results
            result_msg = f"Probability: {probabilities[0]:.4f}\n"
            result_msg += f"Classification: {'POSITIVE' if probabilities[0] > 0.5 else 'NEGATIVE'}\n"
            result_msg += f"Goodness per layer: {[f'{g[0]:.4f}' for g in goodness_values]}"

            self.test_result_label.config(text=result_msg)
            self.log_message(f"\nTest Input: {input_str}")
            self.log_message(result_msg)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to test input:\n{str(e)}")

    def check_queue(self):
        """Check message queue for updates from training thread"""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()

                if msg_type == "log":
                    self.log_message(msg_data)
                elif msg_type == "progress":
                    self.progress_var.set(msg_data)
                elif msg_type == "status":
                    text, color = msg_data
                    self.status_label.config(text=text, foreground=color)
                elif msg_type == "training_done":
                    self.is_training = False
                    self.train_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.init_button.config(state="normal")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.check_queue)


def main():
    """Main entry point for GUI"""
    root = tk.Tk()
    app = FFGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
