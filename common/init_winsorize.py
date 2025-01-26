def initialize_shape(shape_instance, **kwargs):
    for key, value in kwargs.items():
        setattr(shape_instance, key, value)