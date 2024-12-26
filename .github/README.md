<div id="top"></div>
<br/>
<br/>
<br/>


<p align="center">
  <img src="images/bricksfinder.png">
</p>
<h1 align="center">
    <a href="https://github.com/Armaggheddon/BricksFinder">BricksFinder</a>
</h1>
<p align="center">
    <a href="https://github.com/Armaggheddon/BricksFinder/commits/master">
    <img src="https://img.shields.io/github/last-commit/Armaggheddon/BricksFinder">
    </a>
    <a href="https://github.com/Armaggheddon/BricksFinder">
    <img src="https://img.shields.io/badge/Maintained-yes-green.svg">
    </a>
    <a href="https://github.com/Armaggheddon/BricksFinder/issues">
    <img src="https://img.shields.io/github/issues/Armaggheddon/BricksFinder">
    </a>
    <a href="https://github.com/Armaggheddon/BricksFinder/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Armaggheddon/BricksFinder">
    </a>
    <a target="_blank" href="https://colab.research.google.com/github/Armaggheddon/BricksFinder/blob/main/live_demo/live_demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p>
<p align="center">
    Your missing LEGO piece, now just a search away! 🔍🧩
    <br/>
    <br/>
    <a href="https://github.com/Armaggheddon/BricksFinder/issues">Report Bug</a>
    •
    <a href="https://github.com/Armaggheddon/BricksFinder/issues">Request Feature</a>
</p>

---

BricksFinder is a fun project that combines the power of AI with the magic of LEGO! 🧱✨ It leverages a custom dataset, a fine-tuned CLIP model, and a user-friendly Web UI to help you search for LEGO minifigures and bricks using either text or images.

Whether you’re a fan of LEGO or just someone trying to find that missing brick, BricksFinder is here to make your LEGO dreams a reality!

![web_ui](images/webui_demo.webp)

## Features ✨
BricksFinder combines the magic of LEGO with cutting-edge AI to offer:

- **Custom LEGO Datasets 🧱:** A dataset of LEGO minifigures (completed) and an upcoming dataset for LEGO bricks.
- **AI-Powered Search 🔍:** Fine-tuned CLIP models enabling intuitive searches via text or images.
- **User-Friendly Web UI 🌐:** Search for your favorite LEGO pieces with ease, whether you're browsing by description or uploading a photo.
- **Live Demo on Colab 🚀:** Test the functionality instantly using the interactive Google Colab demo!

Discover. Search. Build. BricksFinder makes your LEGO journey smarter and more fun! 🚀✨


## Datasets 🧱
BricksFinder relies on carefully crafted datasets to power its search capabilities:

- **Minifigure Dataset:** A comprehensive dataset of LEGO minifigures. It includes images with the Rebrickable caption and a long caption generated using Gemini-1.5-Flash(002). The dataset is available on [HuggingFace datasets](https://huggingface.co/datasets/armaggheddon97/lego_minifigure_captions). More details can be found in the dedicated [README](../datasets/lego_minifigures_captions/README.md).
- **Brick Dataset (Coming soon):** A dataset for LEGO bricks is currently in progress and will soon be added to the project.


A massive shoutout to the [Rebrickable](https://rebrickable.com/) team for providing the data and images for the LEGO minifigures and bricks!


## Model Fine-Tuning 🧠
BricksFinder takes advantage of two fine-tuned CLIP model (ViT-B/32) to power its search capabilities:

- `clip-vit-base-patch32_lego-minifigure`: The model has been fine-tuned on the LEGO minifigure dataset, available on [HuggingFace](https://huggingface.co/armaggheddon97/clip-vit-base-patch32_lego-minifigure).
- `clip-vit-base-patch32_lego-brick`: Fine-tuning for the LEGO brick dataset is planned and will be added once the dataset is complete.

You can also see the fine-tuned models in action via the Colab live demo! 🚀 <a target="_blank" href="https://colab.research.google.com/github/Armaggheddon/BricksFinder/blob/main/live_demo/live_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Web UI 🌐
The BricksFinder Web UI is ready to go, making it easier than ever to search for LEGO pieces! 🎮✨

- **Search by Text or Image:** Whether you’re describing a piece or uploading an image, find your LEGO minifigure in a flash!
- **Smooth and Fun Experience:** Built with Gradio, the Web UI is designed to be intuitive and user-friendly, ensuring that your LEGO search is a breeze.
- **Brick Dataset (Coming Soon):** While the minifigure dataset is fully supported, the brick index is currently in development. Stay tuned for updates when the brick search functionality goes live! 🚀


Give it a try on the live demo and experience the magic of LEGO search firsthand! 🧱✨ Make sure to select the GPU runtime otherwise you will be waiting for a while ⌚ <a target="_blank" href="https://colab.research.google.com/github/Armaggheddon/BricksFinder/blob/main/live_demo/live_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

> [!NOTE]
> The first startup of the Web UI may take a few minutes depending on the hardware used. On colab, using the free Tesla T4 GPU, it takes around 5 minutes for the minifigure dataset. The brick dataset, when ready, will take longer due to the larger size of the dataset. The required models and datasets are also downloaded during the first startup.

> [!TIP]
> On Colab, unless you save the environment, the data will be lost when closing the notebook. If you plan to reuse the application, make sure to save the environment or download the required files.


## Installation and Usage ⚙️
BricksFinder is designed to be easy to use and accessible to everyone! 🚀 it uses a Docker container at its core and uses Docker Compose for easier setup. To get BricksFinder up and running locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Armaggheddon/BricksFinder.git
   cd BricksFinder
   ```
2. **Build and Run the Docker Container:**
   Two containers are available depending on the available hardware in the `bricks_finder` folder:
    - `docker-compose.yml` for CPU-only
        ```bash
        docker compose up --build
        ```
    - `gpu-docker-compose.yml` if you have an NVIDIA GPU
        ```bash
        docker compose -f gpu-docker-compose.yml up --build
        ```

3. **Access the Web UI:**
    Once the container is up and running, and all the required models and datasets are downloaded and the index has been built, you can access the Web UI at `http://localhost:8000/`.


> [!NOTE]
> As for the Gradio live demo, the first startup may take a few minutes depending on the hardware used and the available internet speed. However when run locally, this will happen only the first time a specific dataset is used and subsequent runs will use the cached data.

> [!TIP]
> When using the GPU container, make sure to have both the latest Nvidia drivers (>= 560 or later) and the latest version of the Nvidia Container Toolkit installed. For more information, refer to the [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## Future Work 🚧
BricksFinder is just getting started, and there's a lot more to come! Here’s what’s on the horizon:

- **Complete the LEGO Brick Dataset 🧱:** The dataset for LEGO bricks is currently in progress and will soon be added, expanding the search functionality.
- **Fine-Tuning for LEGO Bricks 🔍:** Once the brick dataset is complete, we’ll fine-tune the CLIP model for bricks, enabling seamless search by image or text.


## 🤝 Contributing
We’d love to see your contributions! Found a bug? Have a feature idea? Open an issue or submit a pull request. Let’s build something awesome together! 💪


## 📄 License
This project is licensed under the MIT License, so feel free to use it, modify it, and share it. 🎉
