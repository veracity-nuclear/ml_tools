class StatusBar():
    """ A simple class for plotting a status bar to the screen

    Parameters
    ----------
    total_count : int
        The total tick count of the status bar
    length : int
        The length of the status to be printed
    """

    def __init__(self, total_count: int, length: int = 50) -> None:
        self._total = total_count
        self._length = length

    def update(self, i: int) -> None:
        """ Updates the printed status bar with the current "tick count"

        Parameters
        ----------
        i : int
            The current tick count
        """
        frac = (i + 1) / self._total
        completed = int(frac * self._length)
        remaining = self._length - completed
        print(f"[{'|' * completed}{'.' * remaining}] {frac * 100:.1f}%", end='\r')

    def finalize(self) -> None:
        """ Prints the final full bar
        """
        print(f"[{'|' * self._length}] 100.0%")
