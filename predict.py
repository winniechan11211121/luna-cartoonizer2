import torch
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_single_file(
            "models/luna_style_training-10.safetensors",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

    def predict(self,
                prompt: str = Input(description="Prompt to generate image"),
                seed: int = Input(default=42, description="Random seed"),
    ) -> Path:
        generator = torch.manual_seed(seed)
        image = self.pipe(prompt, generator=generator).images[0]
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
