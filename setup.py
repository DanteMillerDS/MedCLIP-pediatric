from setuptools import setup

setup(
    packages=['data_loader', 'fine_tune', 'visualize', 'zero_shot'],
    scripts=['scripts/t_clip.py', 'scripts/t_medclip.py', 'scripts/zs_clip_base.py', 'scripts/zs_medclip_base.py', 'scripts/zs_clip_finetune.py', 'scripts/zs_medclip_finetune.py'] 
)