from .env_utils import is_fbcode


def get_production_shapes(op_name, op_type, shuffle_shapes=False):
    """Gets a list of Softmax shapes for benchmarking"""
    if is_fbcode():
        from .fb.durin_data import productionDataLoader

        return [
            shape
            for shape in productionDataLoader.get_shapes_from_frozen_durin(
                op_name, op_type, shuffle_shapes
            )
        ]
