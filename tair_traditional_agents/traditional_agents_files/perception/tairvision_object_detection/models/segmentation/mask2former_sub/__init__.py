try:
    from .pixel_decoder import MSDeformAttnPixelDecoder, ShapeSpec
    from .multi_scale_decoder import MultiScaleMaskedTransformerDecoder
except:
    print("MultiScaleDeformableAttention is not imported")

