from setuptools import setup,find_packages

setup(
    name='MedCLIP-pediatric',  # Ensure this line is present
    version='0.1.0',
    packages=find_packages(),
    scripts=['scripts/t_clip.py', 'scripts/t_medclip.py', 'scripts/zs_clip_base.py', 'scripts/zs_medclip_base.py', 'scripts/zs_clip_finetune.py', 'scripts/zs_medclip_finetune.py'] 

)
