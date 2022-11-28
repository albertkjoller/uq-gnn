
import glob, os
from PIL import Image

# Found here: https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
def make_gif(img_dir, filename, duration=150):
    frames = [Image.open(image) for image in glob.glob(f"{img_dir}/*.png")]
    epoch_frame = [int(image.split("\\")[-1].split(".")[0]) for image in sorted(glob.glob(f"{img_dir}/*.png"))]
    frames = list(list(zip(*sorted(zip(epoch_frame, frames), key=lambda x: x[0])))[1])
    frame_one = frames[0]
    for _ in range(20):
        frames.insert(0, frame_one)
    cwd = os.getcwd()
    os.chdir(img_dir)
    frame_one.save(f"../{filename}", format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)

    os.chdir(cwd)


if __name__ == '__main__':

    experiment_name = 'REPRODUCTION_debug'

    # when running with args
    make_gif(f"../results/{experiment_name}/ALEATORIC", 'aleatoric.gif', duration=150)
    make_gif(f"../results/{experiment_name}/EPISTEMIC", 'epistemic.gif', duration=150)
    make_gif(f"../results/{experiment_name}/PARAMS", 'parameters.gif', duration=150)
    make_gif(f"../results/{experiment_name}/COMBINED_UNCERTAINTIES", "combined.gif", duration=150)