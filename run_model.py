import exact_data
import load_data
import zero_shot
import visualize
if __name__ == "__main__":
    # Runs exact data script
    
    exact_data.mount_and_process()
    medical_type = 'ucsd'  
    model_type = "medclip"
    batch_size = 256
    generators, lengths = load_data.create_loader(medical_type,batch_size,model_type)
    visualize.save_random_images_from_generators(generators,[medical_type,model_type],2)
    zero_shot.run_zero_shot_classification_medclipmodel(medical_type, generators, lengths)
    
    medical_type = 'ori'  
    generators, lengths = load_data.create_loader(medical_type,batch_size,model_type)
    visualize.save_random_images_from_generators(generators,[medical_type,model_type],2)
    zero_shot.run_zero_shot_classification_medclipmodel(medical_type, generators, lengths)
    
    medical_type = 'ucsd'  
    model_type = "clip"
    batch_size = 256
    generators, lengths = load_data.create_loader(medical_type,batch_size,model_type)
    visualize.save_random_images_from_generators(generators,[medical_type,model_type],2)
    zero_shot.run_zero_shot_classification_clipmodel(medical_type, generators, lengths)
    
    medical_type = 'ori'  
    generators, lengths = load_data.create_loader(medical_type,batch_size,model_type)
    visualize.save_random_images_from_generators(generators,[medical_type,model_type],2)
    zero_shot.run_zero_shot_classification_clipmodel(medical_type, generators, lengths)