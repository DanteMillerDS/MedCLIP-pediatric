import exact_data
import load_data
import zero_shot

if __name__ == "__main__":
    # Runs exact data script
    exact_data.mount_and_process()
    medical_type = 'ucsd'  
    batch_size = 256
    train_generator, validation_generator, test_generator, train_length, validation_length, test_length = load_data.create_loader(medical_type,batch_size)
    zero_shot.run_zero_shot_classification(medical_type, batch_size,train_generator, validation_generator, test_generator, train_length, validation_length, test_length)
    medical_type = 'ori'  
    train_generator, validation_generator, test_generator, train_length, validation_length, test_length = load_data.create_loader(medical_type,batch_size)
    zero_shot.run_zero_shot_classification(medical_type, batch_size, train_generator, validation_generator, test_generator, train_length, validation_length, test_length)