//! View state transitions: deck.gl's `LinearInterpolator`, `FlyToInterpolator` and the part of
//! `TransitionManager` that advances a transition each frame.

use math_gl::fly_to::{fly_to_viewport, get_fly_to_duration, FlyToOptions, FlyToProps};

use crate::deck::ViewState;

/// Maps the elapsed fraction of a transition to progress; deck.gl's default is linear.
pub type Easing = fn(f64) -> f64;

/// How the view state moves from start to end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransitionInterpolator {
    /// Interpolate longitude, latitude, zoom, bearing and pitch linearly
    Linear,
    /// Fly along the van Wijk and Nuij path (zoom out, pan, zoom in); bearing and pitch linear
    FlyTo(FlyToOptions),
}

/// Length of a transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransitionDuration {
    Ms(f64),
    /// Computed from the distance by the fly to interpolator (`transitionDuration: 'auto'`);
    /// 300 ms for the linear one
    Auto,
}

/// What a new transition or a gesture does to a running transition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransitionInterruption {
    /// Stop where it is and start the new one from there
    #[default]
    Break,
    /// Jump to the end of the running transition first
    SnapToEnd,
    /// Keep the running transition, drop the new one
    Ignore,
}

/// deck.gl's `transitionDuration`, `transitionInterpolator`, `transitionEasing` and
/// `transitionInterruption`.
#[derive(Clone, Copy, Debug)]
pub struct TransitionProps {
    pub duration: TransitionDuration,
    pub interpolator: TransitionInterpolator,
    pub easing: Easing,
    pub interruption: TransitionInterruption,
}

impl TransitionProps {
    /// A fly to with the default curve and speed and an automatic duration.
    pub fn fly_to() -> Self {
        Self {
            duration: TransitionDuration::Auto,
            interpolator: TransitionInterpolator::FlyTo(FlyToOptions::default()),
            easing: linear,
            interruption: TransitionInterruption::Break,
        }
    }

    /// A linear transition of the given length.
    pub fn linear(duration_ms: f64) -> Self {
        Self {
            duration: TransitionDuration::Ms(duration_ms),
            interpolator: TransitionInterpolator::Linear,
            easing: linear,
            interruption: TransitionInterruption::Break,
        }
    }

    pub fn with_duration(mut self, duration_ms: f64) -> Self {
        self.duration = TransitionDuration::Ms(duration_ms);
        self
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    pub fn with_interruption(mut self, interruption: TransitionInterruption) -> Self {
        self.interruption = interruption;
        self
    }
}

pub fn linear(t: f64) -> f64 {
    t
}

/// The ease in and out cubic curve, a common `transitionEasing`.
pub fn ease_in_out_cubic(t: f64) -> f64 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

/// A transition in flight between two view states.
#[derive(Clone, Copy, Debug)]
pub struct ViewStateTransition {
    start: ViewState,
    end: ViewState,
    width: f64,
    height: f64,
    start_time: f64,
    duration: f64,
    props: TransitionProps,
}

impl ViewStateTransition {
    /// Start a transition at `now` (milliseconds). `None` when the duration is zero, in which
    /// case the caller should jump to `end`.
    pub fn new(
        start: ViewState,
        end: ViewState,
        width: f64,
        height: f64,
        props: TransitionProps,
        now: f64,
    ) -> Option<Self> {
        let duration = match (props.duration, props.interpolator) {
            (TransitionDuration::Ms(ms), _) => ms,
            (TransitionDuration::Auto, TransitionInterpolator::FlyTo(opts)) => get_fly_to_duration(
                &fly_props(&start, width, height),
                &fly_props(&end, width, height),
                &opts,
            ),
            (TransitionDuration::Auto, TransitionInterpolator::Linear) => 300.0,
        };
        if duration.is_nan() || duration <= 0.0 {
            return None;
        }
        Some(Self {
            start,
            end,
            width,
            height,
            start_time: now,
            duration,
            props,
        })
    }

    pub fn start(&self) -> ViewState {
        self.start
    }

    pub fn end(&self) -> ViewState {
        self.end
    }

    pub fn interruption(&self) -> TransitionInterruption {
        self.props.interruption
    }

    pub fn duration_ms(&self) -> f64 {
        self.duration
    }

    pub fn is_done(&self, now: f64) -> bool {
        now - self.start_time >= self.duration
    }

    /// The view state at time `now`.
    pub fn at(&self, now: f64) -> ViewState {
        let t = ((now - self.start_time) / self.duration).clamp(0.0, 1.0);
        let t = (self.props.easing)(t).clamp(0.0, 1.0);
        let lerp = |a: f64, b: f64| a + (b - a) * t;
        match self.props.interpolator {
            TransitionInterpolator::Linear => ViewState {
                longitude: lerp(self.start.longitude, self.end.longitude),
                latitude: lerp(self.start.latitude, self.end.latitude),
                zoom: lerp(self.start.zoom, self.end.zoom),
                bearing: lerp(self.start.bearing, self.end.bearing),
                pitch: lerp(self.start.pitch, self.end.pitch),
            },
            TransitionInterpolator::FlyTo(opts) => {
                let [longitude, latitude, zoom] = fly_to_viewport(
                    &fly_props(&self.start, self.width, self.height),
                    &fly_props(&self.end, self.width, self.height),
                    t,
                    &opts,
                );
                ViewState {
                    longitude,
                    latitude,
                    zoom,
                    bearing: lerp(self.start.bearing, self.end.bearing),
                    pitch: lerp(self.start.pitch, self.end.pitch),
                }
            }
        }
    }
}

fn fly_props(view: &ViewState, width: f64, height: f64) -> FlyToProps {
    FlyToProps {
        width,
        height,
        longitude: view.longitude,
        latitude: view.latitude,
        zoom: view.zoom,
    }
}

/// Apply deck.gl's interruption rules: what the current view should be and whether the new
/// transition may start, given a running one.
pub fn interrupt(
    running: Option<&ViewStateTransition>,
    current: ViewState,
    interruption: TransitionInterruption,
) -> (ViewState, bool) {
    match (running, interruption) {
        (None, _) => (current, true),
        (Some(_), TransitionInterruption::Break) => (current, true),
        (Some(t), TransitionInterruption::SnapToEnd) => (t.end(), true),
        (Some(_), TransitionInterruption::Ignore) => (current, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(longitude: f64, zoom: f64, bearing: f64) -> ViewState {
        ViewState {
            longitude,
            latitude: 0.0,
            zoom,
            bearing,
            pitch: 0.0,
        }
    }

    #[test]
    fn linear_transition_moves_every_prop() {
        let t = ViewStateTransition::new(
            view(0.0, 2.0, 0.0),
            view(10.0, 4.0, 90.0),
            100.0,
            100.0,
            TransitionProps::linear(1000.0),
            5000.0,
        )
        .unwrap();
        let mid = t.at(5500.0);
        assert!(
            (mid.longitude - 5.0).abs() < 1e-9
                && (mid.zoom - 3.0).abs() < 1e-9
                && (mid.bearing - 45.0).abs() < 1e-9
        );
        assert!(!t.is_done(5999.0));
        assert!(t.is_done(6000.0));
        assert_eq!(t.at(7000.0), view(10.0, 4.0, 90.0));
        assert_eq!(t.at(1000.0), view(0.0, 2.0, 0.0));
    }

    #[test]
    fn fly_to_has_an_automatic_duration_and_zooms_out_between() {
        let t = ViewStateTransition::new(
            view(0.0, 10.0, 0.0),
            view(60.0, 10.0, 0.0),
            800.0,
            600.0,
            TransitionProps::fly_to(),
            0.0,
        )
        .unwrap();
        assert!(t.duration_ms() > 1000.0);
        let mid = t.at(t.duration_ms() / 2.0);
        assert!(mid.zoom < 10.0 && mid.longitude > 0.0 && mid.longitude < 60.0);
        let end = t.at(t.duration_ms());
        assert!((end.longitude - 60.0).abs() < 1e-6 && (end.zoom - 10.0).abs() < 1e-6);
    }

    #[test]
    fn zero_duration_is_no_transition() {
        assert!(ViewStateTransition::new(
            view(0.0, 1.0, 0.0),
            view(1.0, 1.0, 0.0),
            10.0,
            10.0,
            TransitionProps::linear(0.0),
            0.0
        )
        .is_none());
    }

    #[test]
    fn easing_applies() {
        let props = TransitionProps::linear(100.0).with_easing(ease_in_out_cubic);
        let t = ViewStateTransition::new(view(0.0, 0.0, 0.0), view(100.0, 0.0, 0.0), 10.0, 10.0, props, 0.0)
            .unwrap();
        assert!((t.at(25.0).longitude - 6.25).abs() < 1e-9);
        assert!((t.at(50.0).longitude - 50.0).abs() < 1e-9);
    }
}
