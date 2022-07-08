import pickle


class BaseConfig:
    def __init__(self):
        pass

    def verify(self, ignore_keys = []):
        # Ignore keys contains keys we want to skip during verification

        parameters = vars(self)

        for name, value in parameters.items():
            # Check that value is not None
            # Raise Exception if value is None
            if (name not in ignore_keys) and (value is None):
                raise Exception("Missing value for parameter {} in config {}".format(
                        name,
                        self.__class__.__name__
                ))

            # If object is another config object, call verify on that config as well
            if isinstance(value, BaseConfig):
                value.verify()

        print("Verified config {}. Note: this just checks for missing values!".format(self.__class__.__name__))

    def set_parameter(self, name, value):
        """ Set the value of a parameter in the config.

        This is the safe way to set parameters, as it works regardless if the parameter exists or not.
        """

        setattr(self, name, value)

    def get_parameter(self, name):
        """ Get the value of a parameter. This is equivalent to config.{PARAM_NAME}
        """

        return getattr(self, name)