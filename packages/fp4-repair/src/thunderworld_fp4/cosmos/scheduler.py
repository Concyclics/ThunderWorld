"""Keep the known denoising schedule on the host, without changing solver math."""
from diffusers import UniPCMultistepScheduler


class HostScheduleUniPC(UniPCMultistepScheduler):
    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        # Cosmos uses t.item() to fill the current modality timestep tensors.
        # The schedule is computed by NumPy on CPU already; uploading it only
        # to read it back at each step adds an unnecessary device barrier.
        # Sigmas, solver coefficients and actual model inputs are unchanged.
        return super().set_timesteps(num_inference_steps, device="cpu", **kwargs)
