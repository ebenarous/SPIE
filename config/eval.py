import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # directory for generated samples to evaluate
    config.sample_savedir = "./generated_samples"
    # directory for input images
    config.inputimg_dir = "your_input_data_path"
    # directory for style images
    config.styleimg_dir = "your_style_data_path"
    # name of input element
    config.input_name = "your_input_data_name"
    # name of style element
    config.style_name = "your_style_data_name"
    # Edit entire frame
    config.whole_frame = False
    # directory to save evaluation results
    config.eval_savedir = "./evaluation_results"
    # evaluation batch size
    config.batch_size = 32
    # resolution at which to evaluate the images
    config.data_resize = 256

    return config
