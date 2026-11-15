from model_service.idle import IdleRelease

IDLE_SECONDS = 300


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeClassifier:
    """Only what IdleRelease asks of a classifier: where the model is loaded, and to release it."""

    def __init__(self, device):
        self.device = device
        self.releases = 0

    def metadata(self):
        return {"device": self.device} if self.device else None

    def release(self):
        self.releases += 1


def make(device="cuda:0"):
    clock, classifier = Clock(), FakeClassifier(device)
    return clock, classifier, IdleRelease(classifier, IDLE_SECONDS, clock=clock)


def test_the_model_is_released_once_the_idle_time_has_passed():
    clock, classifier, idle = make()
    clock.now = 299
    assert idle.check() == 1
    assert classifier.releases == 0
    clock.now = 300
    assert idle.check() == IDLE_SECONDS
    assert classifier.releases == 1


def test_a_request_starts_the_idle_time_over():
    clock, classifier, idle = make()
    clock.now = 200
    with idle.in_use():
        clock.now = 250
    clock.now = 500
    assert idle.check() == 50
    assert classifier.releases == 0
    clock.now = 550
    idle.check()
    assert classifier.releases == 1


def test_the_model_is_never_released_during_a_request():
    clock, classifier, idle = make()
    with idle.in_use():
        clock.now = 1000
        assert idle.check() == IDLE_SECONDS
        assert classifier.releases == 0
        with idle.in_use():
            pass
        # A second request ended, but the first is still under way.
        clock.now = 2000
        idle.check()
        assert classifier.releases == 0
    # The idle time counts from the end of the last request, not from its start.
    clock.now = 2299
    idle.check()
    assert classifier.releases == 0
    clock.now = 2300
    idle.check()
    assert classifier.releases == 1


def test_a_released_model_is_released_again_only_after_it_is_used_again():
    clock, classifier, idle = make()
    clock.now = 300
    idle.check()
    clock.now = 5000
    idle.check()
    assert classifier.releases == 1
    with idle.in_use():
        pass
    clock.now = 5300
    idle.check()
    assert classifier.releases == 2


def test_the_model_stays_loaded_on_the_cpu():
    clock, classifier, idle = make("cpu")
    clock.now = 5000
    assert idle.check() == IDLE_SECONDS
    assert classifier.releases == 0


def test_a_model_that_was_never_loaded_is_not_released():
    clock, classifier, idle = make(device=None)
    clock.now = 5000
    idle.check()
    assert classifier.releases == 0
