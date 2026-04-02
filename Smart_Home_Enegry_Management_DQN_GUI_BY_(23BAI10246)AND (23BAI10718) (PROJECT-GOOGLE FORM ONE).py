import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque


# ================= ENVIRONMENT =================

class SmartHomeEnv:

    def __init__(self):

        # 24-hour electricity prices
        self.prices = [
            3,3,3,3,3,3,
            5,5,5,5,
            7,7,7,7,
            9,9,9,9,
            6,6,5,4
        ]

        # Power rating (kW)
        self.devices = {
            "Fan": 1,
            "Light": 0.5,
            "AC": 3,
            "TV": 1.5,
            "Washer": 2
        }

        self.hour = 0


    def reset(self):

        self.hour = 0

        return np.array([self.hour / 23], dtype=np.float32)


    def step(self, actions):

        # Keep hour safe
        self.hour = self.hour % len(self.prices)

        price = self.prices[self.hour]

        cost = 0

        for d in actions:
            cost += actions[d] * self.devices[d] * price


        # Comfort reward (encourage ON)
        comfort = sum(actions.values()) * 2

        reward = -cost + comfort


        # Next hour
        self.hour = (self.hour + 1) % len(self.prices)

        next_state = np.array([self.hour / 23], dtype=np.float32)

        return next_state, reward, cost, price



# ================= DQN MODEL =================

class DQN(nn.Module):

    def __init__(self):

        super(DQN, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(1, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )


    def forward(self, x):

        return self.net(x)



# ================= AGENT =================

class DQNAgent:

    def __init__(self):

        self.gamma = 0.95

        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

        self.lr = 0.001

        self.memory = deque(maxlen=6000)

        self.model = DQN()
        self.target = DQN()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.update_target()



    def update_target(self):

        self.target.load_state_dict(self.model.state_dict())



    def remember(self, s, a, r, s2):

        self.memory.append((s, a, r, s2))



    def act(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, 1)


        with torch.no_grad():

            s = torch.FloatTensor(state)

            q = self.model(s)


        return torch.argmax(q).item()



    def replay(self, batch=64):

        if len(self.memory) < batch:
            return


        batch = random.sample(self.memory, batch)


        states, actions, rewards, next_states = zip(*batch)


        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)

        rewards = torch.FloatTensor(rewards)

        next_states = torch.FloatTensor(next_states)


        q_vals = self.model(states)\
            .gather(1, actions.unsqueeze(1)).squeeze()


        next_q = self.target(next_states).max(1)[0]


        target = rewards + self.gamma * next_q


        loss = nn.MSELoss()(q_vals, target.detach())


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        if self.epsilon > self.eps_min:

            self.epsilon *= self.eps_decay



# ================= GUI =================

class SmartHomeGUI:

    def __init__(self, root):

        self.root = root

        self.root.title("Smart Home DQN System (Final)")

        self.root.geometry("780x800")

        self.root.resizable(False, False)


        self.env = SmartHomeEnv()

        self.agent = DQNAgent()


        self.running = False

        self.cost_history = []



        # ===== TITLE =====

        tk.Label(
            root,
            text="SMART HOME ENERGY MANAGEMENT (DQN)",
            font=("Consolas", 20, "bold"),
            fg="cyan"
        ).pack(pady=10)



        # ===== BUTTONS =====

        btn = tk.Frame(root)
        btn.pack(pady=5)


        tk.Button(
            btn, text="Train AI",
            bg="orange", width=12,
            command=self.train
        ).grid(row=0, column=0, padx=5)


        tk.Button(
            btn, text="Start AI",
            bg="green", fg="white",
            width=12, command=self.start
        ).grid(row=0, column=1, padx=5)


        tk.Button(
            btn, text="Stop",
            bg="red", fg="white",
            width=12, command=self.stop
        ).grid(row=0, column=2, padx=5)


        tk.Button(
            btn, text="Show Graph",
            bg="blue", fg="white",
            width=12, command=self.show_graph
        ).grid(row=0, column=3, padx=5)



        # ===== PROGRESS =====

        self.progress = ttk.Progressbar(
            root, length=720, mode="determinate"
        )

        self.progress.pack(pady=10)



        # ===== DEVICE FRAME =====

        self.frame = tk.Frame(root, bd=2, relief="ridge")
        self.frame.pack(padx=20, pady=10, fill="x")


        self.status = {}
        self.dots = {}
        self.switches = {}


        for d in self.env.devices:

            row = tk.Frame(self.frame)
            row.pack(pady=6, padx=10, fill="x")


            tk.Label(
                row, text=d,
                font=("Consolas", 12),
                width=8
            ).pack(side="left")


            s = tk.Label(
                row, text="OFF",
                fg="red",
                font=("Consolas", 12, "bold"),
                width=4
            )
            s.pack(side="left", padx=5)


            dot = tk.Label(
                row, text="●",
                fg="red",
                font=("Consolas", 14)
            )
            dot.pack(side="left")


            var = tk.IntVar()


            chk = tk.Checkbutton(
                row,
                text="Manual ON",
                variable=var
            )
            chk.pack(side="right", padx=5)


            self.status[d] = s
            self.dots[d] = dot
            self.switches[d] = var



        # ===== INFO =====

        info = tk.Frame(root, bd=2, relief="ridge")
        info.pack(padx=20, pady=10, fill="x")


        self.time_lbl = tk.Label(info, text="Hour : 00:00")
        self.time_lbl.pack(pady=3)


        self.price_lbl = tk.Label(info, text="Price/unit : ₹0")
        self.price_lbl.pack(pady=3)


        self.cost_lbl = tk.Label(
            info, text="Cost : ₹0",
            font=("Consolas", 12, "bold")
        )
        self.cost_lbl.pack(pady=3)


        self.msg = tk.Label(
            root, text="Status: Ready",
            fg="yellow"
        )
        self.msg.pack(pady=5)



    # ================= TRAIN =================

    def train(self):

        self.msg.config(text="Status: Training...")

        EPISODES = 500


        for ep in range(EPISODES):

            state = self.env.reset()


            for _ in range(24):

                action = self.agent.act(state)

                actions = {d: action for d in self.env.devices}


                next_state, reward, cost, price = self.env.step(actions)


                self.agent.remember(
                    state, action, reward, next_state
                )


                self.agent.replay()


                state = next_state


            if ep % 25 == 0:

                self.agent.update_target()


            self.progress["value"] = (ep+1)/EPISODES*100

            self.root.update()


        self.msg.config(text="Status: Training Completed ✅")



    # ================= RUN =================

    def start(self):

        self.running = True

        self.msg.config(text="Status: AI Running 🟢")

        self.loop()



    def stop(self):

        self.running = False

        self.msg.config(text="Status: Stopped 🔴")



    def loop(self):

        if not self.running:
            return


        state = np.array(
            [self.env.hour / 23], dtype=np.float32
        )


        action = self.agent.act(state)

        actions = {}


        # Manual override
        for d in self.env.devices:

            if self.switches[d].get() == 1:
                actions[d] = 1
            else:
                actions[d] = action


        next_state, reward, cost, price = self.env.step(actions)


        self.cost_history.append(cost)


        # Update UI
        for d in self.env.devices:

            if actions[d] == 1:

                self.status[d].config(text="ON", fg="green")
                self.dots[d].config(fg="green")

            else:

                self.status[d].config(text="OFF", fg="red")
                self.dots[d].config(fg="red")


        self.time_lbl.config(
            text=f"Hour : {self.env.hour:02d}:00"
        )

        self.price_lbl.config(
            text=f"Price/unit : ₹{price}"
        )

        self.cost_lbl.config(
            text=f"Cost : ₹{round(cost,2)}"
        )


        self.root.after(1500, self.loop)



    # ================= GRAPH =================

    def show_graph(self):

        if not self.cost_history:

            self.msg.config(text="Run AI first!")

            return


        plt.figure("Energy Cost Graph")

        plt.plot(self.cost_history, marker="o")

        plt.xlabel("Time Step")

        plt.ylabel("Cost (₹)")

        plt.title("DQN Energy Optimization")

        plt.grid(True)

        plt.show()



# ================= MAIN =================

if __name__ == "__main__":

    root = tk.Tk()

    SmartHomeGUI(root)

    root.mainloop()