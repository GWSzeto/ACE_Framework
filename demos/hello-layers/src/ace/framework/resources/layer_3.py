from ace.framework.layer import Layer, LayerSettings


class Layer3(Layer):

    @property
    def settings(self):
        return LayerSettings(
            name="layer_3",
            label="Agent Model",
        )

    # TODO: Add valid status checks.
    def status(self):
        self.log.debug(f"Checking {self.labeled_name} status")
        return self.return_status(True)