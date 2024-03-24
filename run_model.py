import exact_data
import data_loader.create_loader

if __name__ == "__main__":
    # Runs exact data script
    exact_data.mount_and_process()
    medical_type = 'ucsd'  
    batch_size = 32
    train_generator, validation_generator, test_generator = data_loader.create_loader(medical_type,batch_size)
