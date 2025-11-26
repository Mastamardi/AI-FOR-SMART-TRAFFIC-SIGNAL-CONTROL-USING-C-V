import pandas as pd
import matplotlib.pyplot as plt

def plot(log_path: str = "traffic_log.csv"):
    df = pd.read_csv(log_path, parse_dates=["timestamp"])
    if df.empty:
        print("No data to plot.")
        return
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    for lane, g in df.groupby("lane"):
        ax1.step(g["timestamp"], g["vehicle_count"], where="post", label=f"Vehicles {lane}")

    ax2.plot(df["timestamp"], df["assigned_green_sec"], color="black", alpha=0.4, label="Green time (s)")
    ax1.set_ylabel("Vehicle count")
    ax2.set_ylabel("Green time (s)")
    ax1.set_xlabel("Time")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Traffic densities and assigned green times")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot()
