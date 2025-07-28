import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# 1. IMU 데이터 불러오기 (npz: joint별 (T,3) 배열)
def load_imu_npz(arr, joint_names):
    imu = {}
    for j in joint_names:
        imu[j] = arr[j]
    return imu

# 2. gif 프레임 로드
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(gif.copy().convert('RGBA'))
            gif.seek(len(frames))
    except EOFError:
        pass
    duration = gif.info.get('duration', 50)
    return frames, duration

# 3. 시각화 및 gif 저장
def visualize_and_save_gif(gif_frames, duration, imu_path, joint_names, save_path):
    imu = np.load(imu_path, allow_pickle=True)
    acc = load_imu_npz(imu["accel"].item(), joint_names)
    gyro = load_imu_npz(imu["gyro"].item(), joint_names)

    n_frames = min(len(gif_frames), min([acc[j].shape[0] for j in joint_names]))
    interval = duration / 1000.0  # sec

    fig = plt.figure(figsize=(12, 8))
    ax_gif = fig.add_subplot(221)
    ax_acc = fig.add_subplot(223)
    ax_gyro = fig.add_subplot(224)
    ax_gif.axis('off')

    acc_lines = {}
    gyro_lines = {}
    for j in joint_names:
        for i, axis in enumerate(['x', 'y', 'z']):
            acc_lines[(j, axis)], = ax_acc.plot([], [], label=f'{j}-{axis}', alpha=0.5)
            gyro_lines[(j, axis)], = ax_gyro.plot([], [], label=f'{j}-{axis}', alpha=0.5)
    ax_acc.set_title('Accelerometer')
    ax_gyro.set_title('Gyroscope')
    ax_acc.set_xlabel('Frame'); ax_gyro.set_xlabel('Frame')
    ax_acc.legend(fontsize=7, ncol=2, loc='upper right')
    ax_gyro.legend(fontsize=7, ncol=2, loc='upper right')
    img = ax_gif.imshow(gif_frames[0])

    acc_ymax = max([np.abs(acc[j]).max() for j in joint_names])
    gyro_ymax = max([np.abs(gyro[j]).max() for j in joint_names])
    ax_acc.set_ylim(-acc_ymax*1.1, acc_ymax*1.1)
    ax_gyro.set_ylim(-gyro_ymax*1.1, gyro_ymax*1.1)
    ax_acc.set_xlim(0, n_frames)
    ax_gyro.set_xlim(0, n_frames)

    plt.tight_layout()

    def update(frame):
        img.set_data(gif_frames[frame])
        ax_gif.set_title(f'Frame {frame+1}/{n_frames}')
        for j in joint_names:
            for i, axis in enumerate(['x', 'y', 'z']):
                acc_lines[(j, axis)].set_data(np.arange(frame+1), acc[j][:frame+1, i])
                gyro_lines[(j, axis)].set_data(np.arange(frame+1), gyro[j][:frame+1, i])
        return [img] + [acc_lines[l] for l in acc_lines] + [gyro_lines[l] for l in gyro_lines]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=duration, blit=False)
    ani.save(save_path, writer=PillowWriter(fps=int(1000/duration)))

    print(f'최종 결과 gif 저장 완료: {save_path}')

# ======= 실제 사용 =======
def main() -> None:
    joint_names = ['Left_knee', 'Left_ankle', 'Head', 'Left_wrist', 'Spine1', 'Spine3', 'Left_elbow']

    gif_folder = Path("results/gif")
    imu_folder = Path("results/imu")
    output_folder = Path("results/final")

    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    output_folder.mkdir(parents=True, exist_ok=True)

    for name in names:
        gif_path = gif_folder / f"{name}.gif"
        if not gif_path.exists():
            raise FileNotFoundError(f"{gif_path} not found.")
        imu_path = imu_folder / f"{name}.npz"
        if not imu_path.exists():
            raise FileNotFoundError(f"{imu_path} not found.")
        output_path = output_folder / f"{name}.gif"

        gif_frames, duration = load_gif_frames(gif_path)

        visualize_and_save_gif(gif_frames, duration, imu_path, joint_names, output_path)
        print(f"Report saved to {name}")

if __name__ == "__main__":
    main()


