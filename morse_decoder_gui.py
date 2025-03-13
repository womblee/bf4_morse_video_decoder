import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading
import os
from PIL import Image, ImageTk

class MorseCodeDecoderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Morse Code Video Decoder")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.video_path = None
        self.output_path = None
        self.is_processing = False
        self.processing_thread = None
        self.decoded_text = ""
        self.morse_sequence = ""
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.brightness_history = []
        
        # Create GUI components
        self.create_menu()
        self.create_main_frame()
        
        # Morse code dictionary
        self.morse_dict = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
            '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
            '-----': '0', '--..--': ',', '.-.-.-': '.', '..--..': '?',
            '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&',
            '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+',
            '-....-': '-', '..--.-': '_', '.-..-.': '"', '...-..-': '$',
            '.--.-.': '@', '': ' '  # Empty string for space
        }
    
    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Video", command=self.open_video)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_main_frame(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video and controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video display frame
        self.video_frame = ttk.LabelFrame(left_panel, text="Video Preview")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas - Fixed size
        self.canvas = tk.Canvas(self.video_frame, bg="black", width=256, height=256)
        self.canvas.pack(padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.open_btn = ttk.Button(control_frame, text="Open Video", command=self.open_video)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = ttk.Button(control_frame, text="Decode", command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.config(state=tk.DISABLED)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.save_btn = ttk.Button(control_frame, text="Save Results", command=self.save_results)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.config(state=tk.DISABLED)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(left_panel, orient="horizontal", 
                                        length=100, mode="determinate", 
                                        variable=self.progress_var)
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Threshold slider
        threshold_frame = ttk.Frame(left_panel)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(threshold_frame, text="Brightness Threshold:").pack(side=tk.LEFT)
        
        self.threshold_var = tk.IntVar(value=128)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0, to=255, 
                                         orient=tk.HORIZONTAL, variable=self.threshold_var)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(threshold_frame, textvariable=self.threshold_var).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results display
        results_frame = ttk.LabelFrame(right_panel, text="Decoded Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Decoded text
        ttk.Label(results_frame, text="Decoded Text:").pack(anchor=tk.W, padx=5, pady=2)
        self.text_output = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=5)
        self.text_output.pack(fill=tk.X, padx=5, pady=5)
        
        # Morse code
        ttk.Label(results_frame, text="Morse Code:").pack(anchor=tk.W, padx=5, pady=2)
        self.morse_output = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=5)
        self.morse_output.pack(fill=tk.X, padx=5, pady=5)
        
        # Brightness plot
        plot_frame = ttk.LabelFrame(right_panel, text="Signal Analysis")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.plot_canvas = tk.Canvas(plot_frame, bg="black", height=150)
        self.plot_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            
            # Open video and display first frame
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.show_error("Could not open video file")
                return
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.process_btn.config(state=tk.NORMAL)
            else:
                self.show_error("Could not read first frame")
    
    def display_frame(self, frame):
        # Always resize to 256x256
        frame = cv2.resize(frame, (256, 256))
        
        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        
        # Update canvas with fixed dimensions
        self.canvas.config(width=256, height=256)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def start_processing(self):
        if not self.video_path or self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.open_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        
        # Clear previous results
        self.text_output.delete(1.0, tk.END)
        self.morse_output.delete(1.0, tk.END)
        self.plot_canvas.delete("all")
        
        # Reset variables
        self.brightness_history = []
        self.current_frame = 0
        self.progress_var.set(0)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.decode_morse_code)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        if self.is_processing:
            self.is_processing = False
            self.status_var.set("Processing stopped by user")
            self.process_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.open_btn.config(state=tk.NORMAL)
    
    def save_results(self):
        if not self.decoded_text:
            self.show_error("No results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f"Decoded Text:\n{self.decoded_text}\n\n")
                    f.write(f"Morse Code:\n{self.morse_sequence}\n")
                self.status_var.set(f"Results saved to: {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"Error saving file: {str(e)}")
    
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About Morse Code Decoder")
        about_window.geometry("400x200")
        about_window.resizable(False, False)
        
        ttk.Label(about_window, text="Morse Code Video Decoder", font=("Helvetica", 14, "bold")).pack(pady=10)
        ttk.Label(about_window, text="A tool to decode Morse code from flashing lights in videos").pack()
        ttk.Label(about_window, text="Version 1.0").pack(pady=10)
        
        ttk.Button(about_window, text="Close", command=about_window.destroy).pack(pady=10)
    
    def show_error(self, message):
        self.status_var.set(f"Error: {message}")
        tk.messagebox.showerror("Error", message)
    
    def decode_morse_code(self):
        """Decode Morse code from video in a background thread"""
        self.status_var.set("Processing video...")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.root.after(0, lambda: self.show_error("Could not open video"))
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust processing based on frame rate
        frame_skip = max(1, int(fps / 15))  # Process at most 15 fps
        
        # Parameters for Morse code detection
        min_signal_duration = max(1, int(fps * 0.05))  # Minimum duration for a signal (0.05 sec)
        brightness_values = []
        
        # Define region of interest (ROI) for detecting the light
        roi_size = min(width, height) // 4
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # First pass: collect brightness values for threshold calculation
        if self.is_processing:
            frame_index = 0
            while cap.isOpened() and self.is_processing and frame_index < min(1000, frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_index % frame_skip != 0:
                    frame_index += 1
                    continue
                
                # Extract region of interest
                roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                brightness = np.mean(roi)
                brightness_values.append(brightness)
                
                frame_index += 1
            
            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate optimal threshold using histogram analysis
        if brightness_values:
            # Use kmeans to find two clusters (on and off states)
            brightness_array = np.array(brightness_values).reshape(-1, 1)
            if len(np.unique(brightness_array)) > 1:  # Ensure we have variation
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(brightness_array.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                threshold = float(np.mean(centers))
            else:
                threshold = float(self.threshold_var.get())
        else:
            threshold = float(self.threshold_var.get())
        
        # Update threshold slider
        self.root.after(0, lambda: self.threshold_var.set(int(threshold)))
        
        # Process frames variables
        last_state = False
        last_change_frame = 0
        frame_index = 0
        light_changes = []
        
        # Create brightness plot image
        max_plot_width = min(1000, frame_count // frame_skip + 1)
        plot_img = np.zeros((150, max_plot_width, 3), dtype=np.uint8)
        plot_index = 0
        
        # Second pass: detect light changes with fixed threshold
        while cap.isOpened() and self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_index + 1) / frame_count * 100
            self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Skip frames according to frame_skip
            if frame_index % frame_skip != 0:
                frame_index += 1
                continue
            
            # Extract region of interest
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Calculate brightness (average pixel value)
            brightness = np.mean(roi)
            
            # Use fixed threshold from analysis
            current_state = brightness > threshold
            
            # Detect change in light state
            if current_state != last_state:
                # Duration of previous state (in frames)
                duration = frame_index - last_change_frame
                
                # Record the change if duration is significant
                if duration >= min_signal_duration:
                    light_changes.append((last_state, duration))
                    last_change_frame = frame_index
                    last_state = current_state
            
            # Update plot with current brightness (if we have room)
            if plot_index < max_plot_width:
                y_pos = int((1 - brightness / 255) * 140) + 5  # Keep within bounds with margin
                y_pos = max(5, min(y_pos, 145))
                plot_img[y_pos, plot_index] = (0, 255, 0) if current_state else (0, 0, 255)
                
                # Draw threshold line
                threshold_y = int((1 - threshold / 255) * 140) + 5
                plot_img[threshold_y, plot_index] = (255, 255, 255)
                
                plot_index += 1
            
            # Display frames more sparingly to avoid overwhelming the UI
            if frame_index % (frame_skip * 5) == 0:
                # Copy frame with ROI marked
                display_frame = frame.copy()
                cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
                
                # Display the state
                state_text = "ON" if current_state else "OFF"
                cv2.putText(display_frame, f"Light: {state_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                self.root.after(0, lambda f=display_frame: self.display_frame(f))
                
                # Update status
                status_text = f"Processing: {frame_index}/{frame_count} frames ({progress:.1f}%)"
                self.root.after(0, lambda s=status_text: self.status_var.set(s))
            
            frame_index += 1
        
        # Process the final state change if needed
        if frame_index > last_change_frame:
            duration = frame_index - last_change_frame
            if duration >= min_signal_duration:
                light_changes.append((last_state, duration))
        
        # After processing all frames, analyze the light changes
        if not self.is_processing:
            cap.release()
            return
        
        if len(light_changes) < 3:
            self.root.after(0, lambda: self.show_error("Not enough light changes detected"))
            cap.release()
            return
        
        # Extract durations for analysis
        on_durations = [duration for state, duration in light_changes if state]
        off_durations = [duration for state, duration in light_changes if not state]
        
        if not on_durations or not off_durations:
            self.root.after(0, lambda: self.show_error("Could not detect clear on/off patterns"))
            cap.release()
            return
        
        # IMPROVED MORSE CODE PARSING ALGORITHM
        # First, normalize all durations to the smallest unit (typically a dot)
        all_durations = on_durations + off_durations
        min_duration = min(all_durations) if all_durations else 1
        
        # Convert durations to normalized units
        normalized_changes = [(state, max(1, round(duration / min_duration))) 
                             for state, duration in light_changes]
        
        # Cluster durations to identify dots, dashes, and different gap types
        on_units = [units for state, units in normalized_changes if state]
        off_units = [units for state, units in normalized_changes if not state]
        
        # Analyze on durations (dots vs dashes)
        if len(set(on_units)) > 1:
            # Use K-means clustering to separate dots and dashes
            on_array = np.array(on_units).reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(on_array, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Sort centers to identify which is dot and which is dash
            centers = centers.flatten()
            sorted_centers = np.sort(centers)
            
            dot_value = float(sorted_centers[0])
            dash_value = float(sorted_centers[1])
            
            # If centers are too close, use standard 1:3 ratio
            if dash_value < 2 * dot_value:
                dash_value = 3 * dot_value
        else:
            # If all on signals are similar, assume they're all dots
            dot_value = float(on_units[0]) if on_units else 1.0
            dash_value = 3 * dot_value
        
        # Analyze off durations (intra-letter gaps vs inter-letter gaps vs word gaps)
        if len(set(off_units)) > 1:
            # Try to identify 2 or 3 different gap types
            k = min(3, len(set(off_units)))
            off_array = np.array(off_units).reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(off_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Sort centers to identify gap types
            centers = centers.flatten()
            sorted_centers = np.sort(centers)
            
            if k == 3:
                intra_letter_gap = float(sorted_centers[0])
                inter_letter_gap = float(sorted_centers[1])
                word_gap = float(sorted_centers[2])
            elif k == 2:
                intra_letter_gap = float(sorted_centers[0])
                inter_letter_gap = float(sorted_centers[1])
                word_gap = 2 * inter_letter_gap
            else:
                intra_letter_gap = float(sorted_centers[0])
                inter_letter_gap = 3 * intra_letter_gap
                word_gap = 7 * intra_letter_gap
        else:
            # If all off signals are similar, use standard ratios
            intra_letter_gap = float(off_units[0]) if off_units else 1.0
            inter_letter_gap = 3 * intra_letter_gap
            word_gap = 7 * intra_letter_gap
        
        # Set thresholds for classification
        dot_dash_threshold = (dot_value + dash_value) / 2
        intra_inter_threshold = (intra_letter_gap + inter_letter_gap) / 2
        inter_word_threshold = (inter_letter_gap + word_gap) / 2
        
        # Build Morse code sequence
        morse_sequence = ""
        current_letter = ""
        
        for i, (state, units) in enumerate(normalized_changes):
            if state:  # Light ON
                if units > dot_dash_threshold:
                    current_letter += "-"  # Dash
                else:
                    current_letter += "."  # Dot
            else:  # Light OFF
                # Check if this is the end of a letter
                if units > intra_inter_threshold:
                    morse_sequence += current_letter + " "
                    current_letter = ""
                    
                    # Check if this is also the end of a word
                    if units > inter_word_threshold:
                        morse_sequence += "/ "  # Word separator
        
        # Add the last letter if there is one
        if current_letter:
            morse_sequence += current_letter
        
        # Clean up the morse sequence
        morse_sequence = morse_sequence.strip()
        
        # Convert Morse sequence to text
        decoded_text = ""
        words = morse_sequence.split("/ ")
        for word in words:
            letters = word.strip().split(" ")
            for letter in letters:
                letter = letter.strip()
                if letter in self.morse_dict:
                    decoded_text += self.morse_dict[letter]
                elif letter:  # If not empty
                    decoded_text += "?"  # Unknown symbol
            decoded_text += " "
        
        decoded_text = decoded_text.strip()
        
        # Store results
        self.morse_sequence = morse_sequence
        self.decoded_text = decoded_text
        
        # Update UI with results
        def update_results():
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, decoded_text)
            
            self.morse_output.delete(1.0, tk.END)
            self.morse_output.insert(tk.END, morse_sequence)
            
            # Display brightness plot
            plot_width = self.plot_canvas.winfo_width()
            plot_height = self.plot_canvas.winfo_height()
            
            # Resize plot image to fit canvas
            if plot_width > 1 and plot_height > 1 and plot_index > 0:
                # Only use the portion of the plot that was filled
                resized_plot = cv2.resize(plot_img[:, :plot_index], (plot_width, plot_height))
                plot_photo = ImageTk.PhotoImage(image=Image.fromarray(resized_plot))
                self.plot_canvas.create_image(0, 0, image=plot_photo, anchor=tk.NW)
                self.plot_canvas.image = plot_photo  # Keep reference
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            
            # Update status
            self.status_var.set(f"Decoding complete: {len(decoded_text)} characters decoded")
        
        self.root.after(0, update_results)
        
        # Reset processing state
        self.is_processing = False
        self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.open_btn.config(state=tk.NORMAL))
        
        # Close video
        cap.release()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MorseCodeDecoderGUI(root)
    root.mainloop()