"""
First Prototype - Grid

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

"""


import tkinter as tk
from stable_baselines3 import DQN, PPO, A2C
from Environment import RobotNavEnv

########################################
# MAIN GUI
########################################

class MainGUI:
    """
    The main application window:
      - Lets you select which model to load (DQN/PPO/A2C)
      - Loads the model from folder
      - Has a small 'pick area' canvas for the user to click & choose a target cell , basically where they 
        want the robot to navigate to
      - Once clicked on start navigations , it opens SimulationWindow when clicked , 
        which shows the environment and animates the robot's navigation
      - The environment is a grid with obstacles, and the robot starts in the center. The obstacles are randomly placed.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("Main Menu - Robot Navigation")

        # Dictionary holding references to loaded models
        self.models = {
            "DQN": None,
            "PPO": None,
            "A2C": None
        }

        # Frame for model selection
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(side=tk.TOP, pady=10)

        tk.Label(self.top_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar(value="DQN")
        self.model_menu = tk.OptionMenu(self.top_frame, self.model_var, "DQN", "PPO", "A2C")
        self.model_menu.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(self.top_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(master, text="No model loaded.")
        self.status_label.pack()

        # Canvas for picking target location
        self.pick_canvas_size = 500  # px
        self.grid_size_main = 20     # We'll assume a 10x10 "pick area"
        self.pick_canvas = tk.Canvas(master, width=self.pick_canvas_size,
                                     height=self.pick_canvas_size, bg="white")
        self.pick_canvas.pack(pady=10)

        # Draw a simple 10x10 grid on this canvas
        self.draw_pick_area()
        
        # Bind click event
        self.pick_canvas.bind("<Button-1>", self.on_canvas_click)

    def draw_pick_area(self):
        """ Draw a 10x10 grid so the user can see where they're clicking. """
        cell_px = self.pick_canvas_size // self.grid_size_main
        for i in range(self.grid_size_main+1):
            # Vertical lines
            self.pick_canvas.create_line(i*cell_px, 0,
                                         i*cell_px, self.pick_canvas_size,
                                         fill="gray")
            # Horizontal lines
            self.pick_canvas.create_line(0, i*cell_px,
                                         self.pick_canvas_size, i*cell_px,
                                         fill="gray")

        # Draw a small circle in the center to represent the robot's default start
        center_x = (self.grid_size_main//2)*cell_px + cell_px//2
        center_y = (self.grid_size_main//2)*cell_px + cell_px//2
        self.pick_canvas.create_oval(center_x-5, center_y-5,
                                     center_x+5, center_y+5,
                                     fill="blue")

    def load_model(self):
        """Load the selected model from disk."""
        chosen_model = self.model_var.get()
        try:
            if chosen_model == "DQN":
                self.models[chosen_model] = DQN.load("models/dqn_robot")
            elif chosen_model == "PPO":
                self.models[chosen_model] = PPO.load("models/ppo_robot")
            elif chosen_model == "A2C":
                self.models[chosen_model] = A2C.load("models/a2c_robot")
            self.status_label.config(text=f"{chosen_model} model loaded!")
        except Exception as e:
            self.status_label.config(text=f"Error loading {chosen_model}: {e}")

    def on_canvas_click(self, event):
        """User clicked a point on the 'pick area' -> open SimulationWindow."""
        chosen_model = self.model_var.get()
        model = self.models.get(chosen_model)
        if model is None:
            self.status_label.config(text="Please load a model first.")
            return

        # Convert click (event.x, event.y) to 10x10 grid coordinates
        cell_px = self.pick_canvas_size // self.grid_size_main
        x_cell = event.x // cell_px
        y_cell = event.y // cell_px

        # Open the simulation window, pass the chosen model + target
        sim_win = SimulationWindow(
            self.master,
            model=model,
            target_x=x_cell,     # user-chosen x
            target_y=y_cell,     # user-chosen y
            grid_size=20,        # environment size
            obstacle_count=25,   # number of obstacles
            max_steps=200
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()


########################################
# SIMULATION WINDOW
########################################

CELL_SIZE = 40

class SimulationWindow(tk.Toplevel):
    """
    A separate window (child of the main GUI) that:
      - Creates a new RobotNavEnv
      - Sets the user-chosen target (if valid)
      - Displays the environment and a 'Start Navigation' button
      - Animates the robot's movement step by step
      - Tell each step's action and reward in a label , which we could use to actually direct the irl robot's movements
    """
    def __init__(self, master, model, target_x, target_y, 
                 grid_size=10, obstacle_count=10, max_steps=200):
        super().__init__(master)
        self.title("Robot Navigation Simulation")

        # Create environment
        self.env = RobotNavEnv(grid_size=grid_size,
                               obstacle_count=obstacle_count,
                               max_steps=max_steps)
        self.obs = self.env.reset()

        self.model = model
        self.done = False

        # Override the environment's random target with the user-chosen coordinates, if valid
        if (0 <= target_x < self.env.grid_size and
            0 <= target_y < self.env.grid_size and
            (target_x, target_y) not in self.env.obstacles):
            self.env.target_pos = (target_x, target_y)
        else:
            print("[SimulationWindow] WARNING: user-chosen target is invalid or on an obstacle."
                  " Using random target from env instead.")

        # Create canvas
        w = self.env.grid_size * CELL_SIZE
        h = self.env.grid_size * CELL_SIZE
        self.canvas = tk.Canvas(self, width=w, height=h, bg="white")
        self.canvas.pack()

        # Action label
        self.action_label = tk.Label(self, text="Action: None")
        self.action_label.pack()

        # Start button
        self.start_button = tk.Button(self, text="Start Navigation", command=self.run_navigation)
        self.start_button.pack(pady=5)

        # Draw initial environment
        self.draw_env()

    def run_navigation(self):
        """Perform one step in the environment and schedule the next step until done."""
        if self.done:
            return

        # Predict action from model
        action, _states = self.model.predict(self.obs, deterministic=True)
        action = int(action)  # or action = action.item()

        self.obs, reward, done, info = self.env.step(action)
        self.done = done

        # Update action label
        action_text = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right",
            4: "Stay"
        }.get(action, "Unknown")
        self.action_label.config(text=f"Action: {action_text}, Reward: {reward:.2f}")

        # Redraw environment
        self.draw_env()

        if not self.done:
            # Schedule the next step after 300 ms
            self.after(300, self.run_navigation)
        else:
            self.action_label.config(text=f"Action: {action_text} (Finished)")

    def draw_env(self):
        self.canvas.delete("all")

        # Draw obstacles (black)
        for (ox, oy) in self.env.obstacles:
            self.canvas.create_rectangle(
                ox * CELL_SIZE, oy * CELL_SIZE,
                (ox+1)*CELL_SIZE, (oy+1)*CELL_SIZE,
                fill="black"
            )

        # Draw target (red)
        tx, ty = self.env.target_pos
        self.canvas.create_rectangle(
            tx * CELL_SIZE, ty * CELL_SIZE,
            (tx+1)*CELL_SIZE, (ty+1)*CELL_SIZE,
            fill="red"
        )

        # Draw robot (blue circle)
        rx, ry = self.env.robot_pos
        self.canvas.create_oval(
            rx * CELL_SIZE + 5, ry * CELL_SIZE + 5,
            (rx+1)*CELL_SIZE - 5, (ry+1)*CELL_SIZE - 5,
            fill="blue"
        )