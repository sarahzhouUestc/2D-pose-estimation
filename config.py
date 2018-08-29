class CONFIG(object):

    """
    General settings
    """
    input_size = 368
    heatmap_size = 46                   
    cpm_stages = 6
    hm_gaussian_variance = 1.0           
    center_radius = 21
    cmap_radius = 40
    num_of_joints = 16
    color_channel = 'RGB'

    base_path = "/home/administrator/PengXiao/tensorflowtest/2D-pose"
    image_dir = "/home/administrator/diskb/PengXiao/code/2D-pose/dataset/MPI/images/"
    model_dir = "/home/administrator/diskb/PengXiao/code/2D-pose/ckpt_models"
    middle_output = "/home/administrator/diskb/PengXiao/code/2D-pose/middle"
    model_name = "cpm.ckpt"


    training_iters = 50000
    validation_iters = 1000
    model_save_iters = 200

