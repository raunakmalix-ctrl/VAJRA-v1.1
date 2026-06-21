class BaseEngine:
    """All engines follow this interface."""

    def load(self):
        """Load model(s) into memory."""
        pass

    def run(self, *args, **kwargs):
        """Run inference. Returns an output file path."""
        pass

    def unload(self):
        """Free GPU memory."""
        pass
