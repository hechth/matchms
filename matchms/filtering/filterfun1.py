def filterfun1(spectrum):

    spectrum = spectrum.clone()
    spectrum.metadata["some_property1"] = "some_value1"
    return spectrum
