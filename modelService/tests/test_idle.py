from threading import Event, Thread

from model_service.idle import IdleRelease

IDLE_SECONDS = 300


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeModel:
    """Only what IdleRelease asks of a model: to release it."""

    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


def make():
    clock, classifier, rewriter = Clock(), FakeModel(), FakeModel()
    return clock, classifier, rewriter, IdleRelease((classifier, rewriter), IDLE_SECONDS, clock=clock)


def used(idle, model):
    """One request for the model, which loads it."""
    with idle.in_use(model):
        pass


def test_the_model_is_released_once_the_idle_time_has_passed():
    clock, classifier, _, idle = make()
    used(idle, classifier)
    clock.now = 299
    assert idle.check() == 1
    assert classifier.releases == 0
    clock.now = 300
    assert idle.check() == IDLE_SECONDS
    assert classifier.releases == 1


def test_a_request_starts_the_idle_time_over():
    clock, classifier, _, idle = make()
    clock.now = 200
    with idle.in_use(classifier):
        clock.now = 250
    clock.now = 500
    assert idle.check() == 50
    assert classifier.releases == 0
    clock.now = 550
    idle.check()
    assert classifier.releases == 1


def test_the_model_is_never_released_during_a_request():
    clock, classifier, _, idle = make()
    with idle.in_use(classifier):
        clock.now = 1000
        assert idle.check() == IDLE_SECONDS
        assert classifier.releases == 0
        with idle.in_use(classifier):
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
    clock, classifier, _, idle = make()
    used(idle, classifier)
    clock.now = 300
    idle.check()
    clock.now = 5000
    idle.check()
    assert classifier.releases == 1
    used(idle, classifier)
    clock.now = 5300
    idle.check()
    assert classifier.releases == 2


def test_a_model_that_was_never_used_is_not_released():
    clock, classifier, rewriter, idle = make()
    clock.now = 5000
    idle.check()
    assert (classifier.releases, rewriter.releases) == (0, 0)


def test_a_request_for_one_model_releases_the_other_and_only_once():
    clock, classifier, rewriter, idle = make()
    used(idle, classifier)
    used(idle, rewriter)
    assert (classifier.releases, rewriter.releases) == (1, 0)
    # The classifier is already released, so a second rewrite releases nothing.
    used(idle, rewriter)
    assert (classifier.releases, rewriter.releases) == (1, 0)
    # The next request for the released model takes the memory back from the other.
    used(idle, classifier)
    assert (classifier.releases, rewriter.releases) == (1, 1)
    # Only the model used last is loaded, so only it is released when the service goes idle.
    clock.now = 300
    idle.check()
    assert (classifier.releases, rewriter.releases) == (2, 1)


def test_a_request_waits_while_the_other_model_has_one_under_way():
    _, classifier, rewriter, idle = make()
    started, entered = Event(), Event()

    def rewrite():
        started.set()
        with idle.in_use(rewriter):
            entered.set()

    with idle.in_use(classifier):
        thread = Thread(target=rewrite)
        thread.start()
        started.wait(5)
        # The rewrite must not start, and the classifier must not be released, in the middle of this request.
        assert not entered.wait(0.2)
        assert classifier.releases == 0
    assert entered.wait(5)
    thread.join(5)
    assert (classifier.releases, rewriter.releases) == (1, 0)
