import os
import re
import csv
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


NON_OPTIMIZED_DIR = "records"
OPTIMIZED_DIR = "records_opt"

NON_OPTIMIZED_BASE_PATH = os.path.join(
    "rm_publisher",
    NON_OPTIMIZED_DIR,
)

OPTIMIZED_BASE_PATH = os.path.join(
    "rm_publisher",
    OPTIMIZED_DIR,
)

# Cross-model thesis plots.
THESIS_PLOTS_PATH = os.path.join(
    "rm_publisher",
    "physical_robot_thesis_plots",
)


# "mae" or "mse"
METRIC = "mse"

# Physical evaluation length.
MAX_STEPS = 100

# Real-robot control timestep [s].
DT = 0.02

# 95% percentile confidence interval settings.
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 42

SAVE_PLOTS = True
SHOW_PLOTS = False


TRAJECTORY_ORDER = [
    "changing_velocities",
    "circle_small_radius",
    "connected_u_curves_5",
    "connected_u_curves_8",
]

TRAJECTORY_DISPLAY_NAMES = {
    "changing_velocities": "Changing Velocities",
    "circle_small_radius": "Small Radius Circle",
    "connected_u_curves_5": "Connected U-Curves (5)",
    "connected_u_curves_8": "Connected U-Curves (8)",
}


def trajectory_display_name(name):
    """
    Convert trajectory folder names to publication-friendly labels.
    """
    return TRAJECTORY_DISPLAY_NAMES.get(
        name,
        name.replace("_", " ").title(),
    )


def ordered_available_trajectories(*aggregated_sets):
    """
    Return all trajectories in the preferred thesis order.
    Any unknown trajectories are appended alphabetically.
    """
    available = set()

    for aggregated in aggregated_sets:
        available.update(
            trajectory
            for trajectory, _ in aggregated.keys()
        )

    ordered = [
        trajectory
        for trajectory in TRAJECTORY_ORDER
        if trajectory in available
    ]

    remaining = sorted(
        available - set(ordered)
    )

    ordered.extend(remaining)

    return ordered



def vector_error_per_step(y_true, y_pred, metric="mae"):
    """
    Returns one vector-error value for every timestep.

    Position:
        XY -> norm([dx, dy]) for MAE
        XY -> dx^2 + dy^2 for MSE

    Velocity:
        [vx, vy, omega] -> vector norm for MAE
        [vx, vy, omega] -> sum of squared component errors for MSE
    """
    diff = y_true - y_pred

    if metric == "mae":
        return np.linalg.norm(
            diff,
            axis=1,
        )

    elif metric == "mse":
        return np.sum(
            diff ** 2,
            axis=1,
        )

    else:
        raise ValueError(
            "metric must be 'mae' or 'mse'"
        )


def per_dim_error_per_step(
    y_true,
    y_pred,
    metric="mae",
):
    """
    Returns one error value per timestep and dimension.

    Shape:
        (num_steps, num_dimensions)
    """
    diff = y_true - y_pred

    if metric == "mae":
        return np.abs(diff)

    elif metric == "mse":
        return diff ** 2

    else:
        raise ValueError(
            "metric must be 'mae' or 'mse'"
        )


def bootstrap_mean_ci(
    values,
    confidence=CONFIDENCE_LEVEL,
    n_bootstrap=N_BOOTSTRAP,
    random_state=BOOTSTRAP_SEED,
):
    """
    Calculates the arithmetic mean and percentile-bootstrap
    confidence interval for the mean.

    IMPORTANT:
        The values passed to this function are timestep-level
        errors.

        All timesteps from all repetitions and seeds belonging
        to one controller are pooled before bootstrapping.
    """
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean_value = float(
        np.mean(values)
    )

    if len(values) == 1:
        return (
            mean_value,
            mean_value,
            mean_value,
        )

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    n = len(values)

    batch_size = 250
    index = 0

    while index < n_bootstrap:

        current_batch_size = min(
            batch_size,
            n_bootstrap - index,
        )

        sample_indices = rng.integers(
            0,
            n,
            size=(
                current_batch_size,
                n,
            ),
        )

        bootstrap_means[
            index:index + current_batch_size
        ] = np.mean(
            values[sample_indices],
            axis=1,
        )

        index += current_batch_size

    alpha = (
        1.0 - confidence
    ) / 2.0

    ci_low = float(
        np.quantile(
            bootstrap_means,
            alpha,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha,
        )
    )

    return (
        mean_value,
        ci_low,
        ci_high,
    )


def normalize_controller_name(
    controller_name,
):
    """
    Removes only the seed from a controller/model name.

    Examples
    --------
    ik
        -> ik

    offset_42_false
        -> offset_false

    offset_24_false
        -> offset_false

    Racetrack_offset_42_false
        -> Racetrack_offset_false

    Racetrack_full_13_true
        -> Racetrack_full_true
    """

    if controller_name.lower() == "ik":
        return "ik"

    match = re.match(
        r"^(.*)_(\d+)_([^_]+)$",
        controller_name,
    )

    if match:

        prefix = match.group(1)
        suffix = match.group(3)

        return (
            f"{prefix}_{suffix}"
        )

    return controller_name


def controller_display_name(
    controller_name,
):
    """
    Convert internal controller names to thesis labels.

    False -> trained without domain randomization
    True  -> trained with domain randomization
    """

    name = str(
        controller_name
    ).lower()

    if (
        name == "ik"
        or name.endswith("_ik")
    ):
        return "IK"

    is_dr = (
        name.endswith("_true")
        or name.endswith("_rand")
        or name.endswith("_randomized")
        or name.endswith("_dr")
    )

    if "offset" in name:
        label = "Hybrid Offset"

    elif (
        "full" in name
        or "policy" in name
    ):
        label = "Fully Learned"

    else:
        return str(
            controller_name
        )

    if is_dr:
        label += " (DR)"

    return label


def controller_sort_key(
    controller_name,
):
    """
    Consistent controller ordering.
    """

    order = {
        "IK": 0,
        "Hybrid Offset": 1,
        "Hybrid Offset (DR)": 2,
        "Fully Learned": 3,
        "Fully Learned (DR)": 4,
    }

    label = controller_display_name(
        controller_name
    )

    return (
        order.get(
            label,
            99,
        ),
        label,
    )


CONTROLLER_COLORS = {
    "IK": "tab:blue",
    "Hybrid Offset": "tab:orange",
    "Hybrid Offset (DR)": "tab:red",
    "Fully Learned": "tab:green",
    "Fully Learned (DR)": "tab:purple",
}


def controller_color(
    controller_name,
):
    """
    Return the controller color.
    """

    label = controller_display_name(
        controller_name
    )

    return CONTROLLER_COLORS.get(
        label,
        None,
    )


def integrate_body_velocities(
    velocities,
    dt=DT,
):
    """
    Reconstruct a planar trajectory from body-frame velocities.

    Expected velocity columns:
        velocities[:, 0] -> vx [m/s]
        velocities[:, 1] -> vy [m/s]
        velocities[:, 2] -> omega [rad/s]

    Every trajectory is reconstructed in its own normalized local
    frame with the initial pose

        x_0 = 0,
        y_0 = 0,
        theta_0 = 0.

    The body-frame translational velocity is rotated into this
    local frame using the integrated heading before the position
    is advanced. The returned array has N + 1 XY poses for N
    velocity samples.

    IMPORTANT:
        This function is used ONLY for the XY trajectory plots.
        The position and velocity benchmark errors are still
        calculated from the original recorded CSV signals.
    """

    velocities = np.asarray(
        velocities,
        dtype=float,
    )

    if velocities.ndim == 1:
        velocities = np.expand_dims(
            velocities,
            axis=0,
        )

    if len(velocities) == 0:
        return np.zeros(
            (1, 2),
            dtype=float,
        )

    if velocities.shape[1] < 3:
        raise ValueError(
            "Velocity data must contain at least three columns: "
            "vx, vy, omega."
        )

    num_steps = len(
        velocities
    )

    trajectory = np.zeros(
        (num_steps + 1, 2),
        dtype=float,
    )

    theta = 0.0

    for step in range(
        num_steps
    ):

        vx = velocities[
            step, 0
        ]

        vy = velocities[
            step, 1
        ]

        omega = velocities[
            step, 2
        ]

        cos_theta = np.cos(
            theta
        )

        sin_theta = np.sin(
            theta
        )

        vx_world = (
            cos_theta * vx
            - sin_theta * vy
        )

        vy_world = (
            sin_theta * vx
            + cos_theta * vy
        )

        trajectory[
            step + 1, 0
        ] = (
            trajectory[
                step, 0
            ]
            + vx_world * dt
        )

        trajectory[
            step + 1, 1
        ] = (
            trajectory[
                step, 1
            ]
            + vy_world * dt
        )

        theta = (
            theta
            + omega * dt
        )

        theta = (
            theta + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi

    return trajectory


def load_run(
    run_path,
):
    """
    Load one recorded physical benchmark run.

    Expected structure:

        run_xxx/
            actual_positions.csv
            actual_velocities.csv
            reference_positions.csv
            reference_velocities.csv
    """

    required_files = {
        "actual_pos":
            "actual_positions.csv",

        "reference_pos":
            "reference_positions.csv",

        "actual_vel":
            "actual_velocities.csv",

        "reference_vel":
            "reference_velocities.csv",
    }

    paths = {
        key: os.path.join(
            run_path,
            filename,
        )
        for key, filename
        in required_files.items()
    }

    missing_files = [
        file_path
        for file_path
        in paths.values()
        if not os.path.isfile(
            file_path
        )
    ]

    if missing_files:

        print(
            "[WARNING] Skipping incomplete run:"
        )

        print(
            f"          {run_path}"
        )

        for missing_file in missing_files:

            print(
                f"          Missing: "
                f"{os.path.basename(missing_file)}"
            )

        return None

    try:

        actual_pos = np.loadtxt(
            paths["actual_pos"],
            delimiter=",",
        )

        ref_pos = np.loadtxt(
            paths["reference_pos"],
            delimiter=",",
        )

        actual_vel = np.loadtxt(
            paths["actual_vel"],
            delimiter=",",
        )

        ref_vel = np.loadtxt(
            paths["reference_vel"],
            delimiter=",",
        )

    except Exception as exc:

        print(
            f"[WARNING] Could not load "
            f"{run_path}: {exc}"
        )

        return None

    actual_pos = np.atleast_2d(
        actual_pos
    )

    ref_pos = np.atleast_2d(
        ref_pos
    )

    actual_vel = np.atleast_2d(
        actual_vel
    )

    ref_vel = np.atleast_2d(
        ref_vel
    )


    n_pos = min(
        MAX_STEPS,
        len(ref_pos),
        len(actual_pos),
    )

    if n_pos == 0:

        print(
            f"[WARNING] Empty position data: "
            f"{run_path}"
        )

        return None

    ref_pos = ref_pos[
        :n_pos
    ]

    actual_pos = actual_pos[
        :n_pos
    ]

    n_vel = min(
        MAX_STEPS,
        len(ref_vel),
        len(actual_vel),
    )

    if n_vel == 0:

        print(
            f"[WARNING] Empty velocity data: "
            f"{run_path}"
        )

        return None

    ref_vel = ref_vel[
        :n_vel
    ]

    actual_vel = actual_vel[
        :n_vel
    ]

    position_xy_step_errors = (
        vector_error_per_step(
            ref_pos[:, :2],
            actual_pos[:, :2],
            metric=METRIC,
        )
    )

    position_per_dim_step_errors = (
        per_dim_error_per_step(
            ref_pos,
            actual_pos,
            metric=METRIC,
        )
    )


    velocity_vector_step_errors = (
        vector_error_per_step(
            ref_vel,
            actual_vel,
            metric=METRIC,
        )
    )

    velocity_per_dim_step_errors = (
        per_dim_error_per_step(
            ref_vel,
            actual_vel,
            metric=METRIC,
        )
    )

    ref_pos_xy = integrate_body_velocities(
        ref_vel,
        dt=DT,
    )

    actual_pos_xy = integrate_body_velocities(
        actual_vel,
        dt=DT,
    )

    return {
        "position_xy_step_errors":
            position_xy_step_errors,

        "position_per_dim_step_errors":
            position_per_dim_step_errors,

        "velocity_vector_step_errors":
            velocity_vector_step_errors,

        "velocity_per_dim_step_errors":
            velocity_per_dim_step_errors,

        "ref_pos_xy":
            ref_pos_xy,

        "actual_pos_xy":
            actual_pos_xy,

        "n_pos":
            n_pos,

        "n_vel":
            n_vel,
    }


def discover_runs(
    base_path,
):
    """
    Search automatically for:

        base_path/
            trajectory/
                controller/
                    run_001/
                    run_002/
                    ...
    """

    runs = []

    if not os.path.isdir(
        base_path
    ):

        raise RuntimeError(
            f"Record directory does not exist: "
            f"{base_path}"
        )

    trajectory_names = sorted(
        name
        for name
        in os.listdir(base_path)
        if os.path.isdir(
            os.path.join(
                base_path,
                name,
            )
        )
        and name != "plots"
    )

    if not trajectory_names:

        raise RuntimeError(
            f"No trajectory directories found "
            f"in {base_path}"
        )

    for trajectory_name in trajectory_names:

        trajectory_path = os.path.join(
            base_path,
            trajectory_name,
        )

        controller_names = sorted(
            name
            for name
            in os.listdir(
                trajectory_path
            )
            if os.path.isdir(
                os.path.join(
                    trajectory_path,
                    name,
                )
            )
        )

        for raw_controller_name in controller_names:

            controller_path = os.path.join(
                trajectory_path,
                raw_controller_name,
            )

            grouped_controller_name = (
                normalize_controller_name(
                    raw_controller_name
                )
            )

            run_names = sorted(
                name
                for name
                in os.listdir(
                    controller_path
                )
                if os.path.isdir(
                    os.path.join(
                        controller_path,
                        name,
                    )
                )
                and name.startswith(
                    "run_"
                )
            )

            for run_name in run_names:

                run_path = os.path.join(
                    controller_path,
                    run_name,
                )

                result = load_run(
                    run_path
                )

                if result is None:
                    continue

                runs.append({
                    "trajectory":
                        trajectory_name,

                    "controller":
                        grouped_controller_name,

                    "raw_controller":
                        raw_controller_name,

                    "run":
                        run_name,

                    "run_path":
                        run_path,

                    **result,
                })

    return runs


def average_trajectory_runs(
    group_runs,
):
    """
    Calculates point-wise mean reference and actual trajectory.
    """

    if not group_runs:
        return None, None

    min_length = min(
        min(
            len(
                run["ref_pos_xy"]
            ),
            len(
                run["actual_pos_xy"]
            ),
        )
        for run in group_runs
    )

    reference_runs = np.stack([
        run["ref_pos_xy"][
            :min_length
        ]
        for run in group_runs
    ])

    actual_runs = np.stack([
        run["actual_pos_xy"][
            :min_length
        ]
        for run in group_runs
    ])

    mean_reference = np.mean(
        reference_runs,
        axis=0,
    )

    mean_actual = np.mean(
        actual_runs,
        axis=0,
    )

    return (
        mean_reference,
        mean_actual,
    )


def median_trajectory_runs(
    group_runs,
):
    """
    Calculates the point-wise median reference and actual trajectory.

    The median is calculated independently for X and Y at every
    timestep across all repetitions belonging to the controller.
    As with the mean trajectory, all runs are truncated to the
    shortest common trajectory length before aggregation.
    """

    if not group_runs:
        return None, None

    min_length = min(
        min(
            len(
                run["ref_pos_xy"]
            ),
            len(
                run["actual_pos_xy"]
            ),
        )
        for run in group_runs
    )

    reference_runs = np.stack([
        run["ref_pos_xy"][
            :min_length
        ]
        for run in group_runs
    ])

    actual_runs = np.stack([
        run["actual_pos_xy"][
            :min_length
        ]
        for run in group_runs
    ])

    median_reference = np.median(
        reference_runs,
        axis=0,
    )

    median_actual = np.median(
        actual_runs,
        axis=0,
    )

    return (
        median_reference,
        median_actual,
    )


def aggregate_results(
    runs,
):
    """
    Group by trajectory + normalized controller.

    All timestep-level errors from repetitions and seeds are
    pooled before calculation of mean and bootstrap CI.
    """

    grouped = defaultdict(
        list
    )

    for run in runs:

        key = (
            run["trajectory"],
            run["controller"],
        )

        grouped[key].append(
            run
        )

    aggregated = {}

    for (
        key,
        group_runs,
    ) in grouped.items():

        position_xy_values = np.concatenate([
            run[
                "position_xy_step_errors"
            ]
            for run in group_runs
        ])

        velocity_values = np.concatenate([
            run[
                "velocity_vector_step_errors"
            ]
            for run in group_runs
        ])

        position_per_dim_values = np.concatenate([
            run[
                "position_per_dim_step_errors"
            ]
            for run in group_runs
        ], axis=0)

        velocity_per_dim_values = np.concatenate([
            run[
                "velocity_per_dim_step_errors"
            ]
            for run in group_runs
        ], axis=0)

        raw_controllers = sorted({
            run[
                "raw_controller"
            ]
            for run in group_runs
        })

        (
            mean_reference,
            mean_actual,
        ) = average_trajectory_runs(
            group_runs
        )

        (
            median_reference,
            median_actual,
        ) = median_trajectory_runs(
            group_runs
        )

        (
            position_xy_mean,
            position_xy_ci_low,
            position_xy_ci_high,
        ) = bootstrap_mean_ci(
            position_xy_values
        )

        (
            velocity_vector_mean,
            velocity_vector_ci_low,
            velocity_vector_ci_high,
        ) = bootstrap_mean_ci(
            velocity_values
        )


        position_per_dim_mean = []
        position_per_dim_ci_low = []
        position_per_dim_ci_high = []

        for dim in range(
            position_per_dim_values.shape[1]
        ):

            (
                mean_value,
                ci_low,
                ci_high,
            ) = bootstrap_mean_ci(
                position_per_dim_values[
                    :, dim
                ]
            )

            position_per_dim_mean.append(
                mean_value
            )

            position_per_dim_ci_low.append(
                ci_low
            )

            position_per_dim_ci_high.append(
                ci_high
            )


        velocity_per_dim_mean = []
        velocity_per_dim_ci_low = []
        velocity_per_dim_ci_high = []

        for dim in range(
            velocity_per_dim_values.shape[1]
        ):

            (
                mean_value,
                ci_low,
                ci_high,
            ) = bootstrap_mean_ci(
                velocity_per_dim_values[
                    :, dim
                ]
            )

            velocity_per_dim_mean.append(
                mean_value
            )

            velocity_per_dim_ci_low.append(
                ci_low
            )

            velocity_per_dim_ci_high.append(
                ci_high
            )

        aggregated[key] = {
            "num_runs":
                len(group_runs),

            "num_position_steps":
                len(
                    position_xy_values
                ),

            "num_velocity_steps":
                len(
                    velocity_values
                ),

            "raw_controllers":
                raw_controllers,

            "position_xy_mean":
                position_xy_mean,

            "position_xy_ci_low":
                position_xy_ci_low,

            "position_xy_ci_high":
                position_xy_ci_high,

            "position_per_dim_mean":
                np.asarray(
                    position_per_dim_mean
                ),

            "position_per_dim_ci_low":
                np.asarray(
                    position_per_dim_ci_low
                ),

            "position_per_dim_ci_high":
                np.asarray(
                    position_per_dim_ci_high
                ),

            "position_xy_step_errors":
                position_xy_values,

            "position_per_dim_step_errors":
                position_per_dim_values,

            "velocity_vector_mean":
                velocity_vector_mean,

            "velocity_vector_ci_low":
                velocity_vector_ci_low,

            "velocity_vector_ci_high":
                velocity_vector_ci_high,

            "velocity_per_dim_mean":
                np.asarray(
                    velocity_per_dim_mean
                ),

            "velocity_per_dim_ci_low":
                np.asarray(
                    velocity_per_dim_ci_low
                ),

            "velocity_per_dim_ci_high":
                np.asarray(
                    velocity_per_dim_ci_high
                ),

            "velocity_vector_step_errors":
                velocity_values,

            "velocity_per_dim_step_errors":
                velocity_per_dim_values,

            "reference_trajectory":
                mean_reference,

            "actual_trajectory":
                mean_actual,

            "reference_trajectory_median":
                median_reference,

            "actual_trajectory_median":
                median_actual,

            "reference_trajectory_runs": [
                run["ref_pos_xy"]
                for run in group_runs
            ],

            "actual_trajectory_runs": [
                run["actual_pos_xy"]
                for run in group_runs
            ],
        }

    return aggregated



def compute_overall_results(
    aggregated,
):
    """
    Compute overall results per controller by pooling all
    trajectories and timesteps.
    """

    controller_results = defaultdict(
        list
    )

    for (
        trajectory,
        controller,
    ), result in aggregated.items():

        controller_results[
            controller
        ].append(
            result
        )

    overall = {}

    for (
        controller,
        results,
    ) in controller_results.items():

        position_values = np.concatenate([
            result[
                "position_xy_step_errors"
            ]
            for result in results
        ])

        velocity_values = np.concatenate([
            result[
                "velocity_vector_step_errors"
            ]
            for result in results
        ])

        position_per_dim_values = np.concatenate([
            result[
                "position_per_dim_step_errors"
            ]
            for result in results
        ], axis=0)

        velocity_per_dim_values = np.concatenate([
            result[
                "velocity_per_dim_step_errors"
            ]
            for result in results
        ], axis=0)

        (
            position_mean,
            position_ci_low,
            position_ci_high,
        ) = bootstrap_mean_ci(
            position_values
        )

        (
            velocity_mean,
            velocity_ci_low,
            velocity_ci_high,
        ) = bootstrap_mean_ci(
            velocity_values
        )

        pos_dim_mean = []
        pos_dim_low = []
        pos_dim_high = []

        for dim in range(
            position_per_dim_values.shape[1]
        ):

            (
                mean_value,
                ci_low,
                ci_high,
            ) = bootstrap_mean_ci(
                position_per_dim_values[
                    :, dim
                ]
            )

            pos_dim_mean.append(
                mean_value
            )

            pos_dim_low.append(
                ci_low
            )

            pos_dim_high.append(
                ci_high
            )

        vel_dim_mean = []
        vel_dim_low = []
        vel_dim_high = []

        for dim in range(
            velocity_per_dim_values.shape[1]
        ):

            (
                mean_value,
                ci_low,
                ci_high,
            ) = bootstrap_mean_ci(
                velocity_per_dim_values[
                    :, dim
                ]
            )

            vel_dim_mean.append(
                mean_value
            )

            vel_dim_low.append(
                ci_low
            )

            vel_dim_high.append(
                ci_high
            )

        overall[controller] = {
            "num_trajectories":
                len(results),

            "num_position_steps":
                len(
                    position_values
                ),

            "num_velocity_steps":
                len(
                    velocity_values
                ),

            "position_xy_mean":
                position_mean,

            "position_xy_ci_low":
                position_ci_low,

            "position_xy_ci_high":
                position_ci_high,

            "position_per_dim_mean":
                np.asarray(
                    pos_dim_mean
                ),

            "position_per_dim_ci_low":
                np.asarray(
                    pos_dim_low
                ),

            "position_per_dim_ci_high":
                np.asarray(
                    pos_dim_high
                ),

            "velocity_vector_mean":
                velocity_mean,

            "velocity_vector_ci_low":
                velocity_ci_low,

            "velocity_vector_ci_high":
                velocity_ci_high,

            "velocity_per_dim_mean":
                np.asarray(
                    vel_dim_mean
                ),

            "velocity_per_dim_ci_low":
                np.asarray(
                    vel_dim_low
                ),

            "velocity_per_dim_ci_high":
                np.asarray(
                    vel_dim_high
                ),
        }

    return overall


def get_result_by_display_name(
    results,
    controller_label,
):
    """
    Find an overall result by display name.
    """

    for (
        controller,
        result,
    ) in results.items():

        if (
            controller_display_name(
                controller
            )
            == controller_label
        ):
            return result

    return None


def get_trajectory_result_by_display_name(
    aggregated,
    trajectory,
    controller_label,
):
    """
    Find one trajectory/controller result by display name.
    """

    for (
        traj,
        controller,
    ), result in aggregated.items():

        if (
            traj == trajectory
            and controller_display_name(
                controller
            ) == controller_label
        ):
            return result

    return None


def error_bar_from_result(
    result,
    metric_key,
    ci_low_key,
    ci_high_key,
):
    """
    Return:
        mean,
        lower asymmetric error,
        upper asymmetric error
    """

    mean_value = result[
        metric_key
    ]

    error_low = max(
        0.0,
        mean_value
        - result[
            ci_low_key
        ],
    )

    error_high = max(
        0.0,
        result[
            ci_high_key
        ]
        - mean_value,
    )

    return (
        mean_value,
        error_low,
        error_high,
    )


def find_trajectory(
    available,
    candidates,
):
    """
    Find the first matching trajectory from several possible names.
    """

    available_lookup = {
        name.lower(): name
        for name in available
    }

    for candidate in candidates:

        if (
            candidate.lower()
            in available_lookup
        ):

            return available_lookup[
                candidate.lower()
            ]

    return None



def print_controller_grouping(
    runs,
):

    controller_mapping = defaultdict(
        set
    )

    for run in runs:

        controller_mapping[
            run["controller"]
        ].add(
            run[
                "raw_controller"
            ]
        )

    print()
    print(
        "=" * 80
    )
    print(
        "CONTROLLER GROUPING"
    )
    print(
        "=" * 80
    )

    controllers = sorted(
        controller_mapping.keys(),
        key=controller_sort_key,
    )

    for controller in controllers:

        print(
            f"{controller_display_name(controller)}:"
        )

        for raw_name in sorted(
            controller_mapping[
                controller
            ]
        ):

            print(
                f"    {raw_name}"
            )

    print()



def print_results(
    aggregated,
    overall,
    dataset_name,
):

    metric_name = METRIC.upper()

    trajectories = (
        ordered_available_trajectories(
            aggregated
        )
    )

    controllers = sorted(
        {
            controller
            for _, controller
            in aggregated.keys()
        },
        key=controller_sort_key,
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"{dataset_name} — "
        f"REAL-ROBOT BENCHMARK — "
        f"{metric_name}"
    )

    print(
        "=" * 80
    )

    for trajectory in trajectories:

        print()
        print(
            "#" * 80
        )

        print(
            f"TRAJECTORY: "
            f"{trajectory_display_name(trajectory)}"
        )

        print(
            "#" * 80
        )

        for controller in controllers:

            key = (
                trajectory,
                controller,
            )

            if key not in aggregated:
                continue

            result = aggregated[
                key
            ]

            print()

            print(
                f"Controller: "
                f"{controller_display_name(controller)}"
            )

            print(
                f"  Policies/seeds: "
                f"{', '.join(result['raw_controllers'])}"
            )

            print(
                f"  Number of runs: "
                f"{result['num_runs']}"
            )

            print()

            print(
                "  Position:"
            )

            print(
                f"    XY {metric_name}: "
                f"{result['position_xy_mean']:.8f} "
                f"[95% CI: "
                f"{result['position_xy_ci_low']:.8f}, "
                f"{result['position_xy_ci_high']:.8f}]"
            )

            print()

            print(
                "  Velocity:"
            )

            print(
                f"    Vector {metric_name}: "
                f"{result['velocity_vector_mean']:.8f} "
                f"[95% CI: "
                f"{result['velocity_vector_ci_low']:.8f}, "
                f"{result['velocity_vector_ci_high']:.8f}]"
            )

    print()
    print(
        "=" * 80
    )
    print(
        "OVERALL RESULTS"
    )
    print(
        "=" * 80
    )

    for controller in sorted(
        overall.keys(),
        key=controller_sort_key,
    ):

        result = overall[
            controller
        ]

        print()

        print(
            f"Controller: "
            f"{controller_display_name(controller)}"
        )

        print(
            f"  Position XY {metric_name}: "
            f"{result['position_xy_mean']:.8f} "
            f"[95% CI: "
            f"{result['position_xy_ci_low']:.8f}, "
            f"{result['position_xy_ci_high']:.8f}]"
        )

        print(
            f"  Velocity Vector {metric_name}: "
            f"{result['velocity_vector_mean']:.8f} "
            f"[95% CI: "
            f"{result['velocity_vector_ci_low']:.8f}, "
            f"{result['velocity_vector_ci_high']:.8f}]"
        )

    print()


def save_summary_csv(
    aggregated,
    overall,
    summary_path,
):

    header = [
        "trajectory",
        "controller",
        "raw_controllers",
        "num_runs",
        "num_position_steps",
        "num_velocity_steps",

        "position_xy_mean",
        "position_xy_ci_low",
        "position_xy_ci_high",

        "position_x_mean",
        "position_x_ci_low",
        "position_x_ci_high",

        "position_y_mean",
        "position_y_ci_low",
        "position_y_ci_high",

        "position_z_mean",
        "position_z_ci_low",
        "position_z_ci_high",

        "velocity_vector_mean",
        "velocity_vector_ci_low",
        "velocity_vector_ci_high",

        "velocity_vx_mean",
        "velocity_vx_ci_low",
        "velocity_vx_ci_high",

        "velocity_vy_mean",
        "velocity_vy_ci_low",
        "velocity_vy_ci_high",

        "velocity_omega_mean",
        "velocity_omega_ci_low",
        "velocity_omega_ci_high",
    ]

    def component(
        values,
        index,
    ):

        if len(values) > index:
            return values[
                index
            ]

        return np.nan

    with open(
        summary_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            header
        )

        for (
            trajectory,
            controller,
        ), result in sorted(
            aggregated.items()
        ):

            pos_mean = result[
                "position_per_dim_mean"
            ]

            pos_low = result[
                "position_per_dim_ci_low"
            ]

            pos_high = result[
                "position_per_dim_ci_high"
            ]

            vel_mean = result[
                "velocity_per_dim_mean"
            ]

            vel_low = result[
                "velocity_per_dim_ci_low"
            ]

            vel_high = result[
                "velocity_per_dim_ci_high"
            ]

            writer.writerow([
                trajectory,

                controller_display_name(
                    controller
                ),

                ";".join(
                    result[
                        "raw_controllers"
                    ]
                ),

                result[
                    "num_runs"
                ],

                result[
                    "num_position_steps"
                ],

                result[
                    "num_velocity_steps"
                ],

                result[
                    "position_xy_mean"
                ],

                result[
                    "position_xy_ci_low"
                ],

                result[
                    "position_xy_ci_high"
                ],

                component(
                    pos_mean,
                    0,
                ),

                component(
                    pos_low,
                    0,
                ),

                component(
                    pos_high,
                    0,
                ),

                component(
                    pos_mean,
                    1,
                ),

                component(
                    pos_low,
                    1,
                ),

                component(
                    pos_high,
                    1,
                ),

                component(
                    pos_mean,
                    2,
                ),

                component(
                    pos_low,
                    2,
                ),

                component(
                    pos_high,
                    2,
                ),

                result[
                    "velocity_vector_mean"
                ],

                result[
                    "velocity_vector_ci_low"
                ],

                result[
                    "velocity_vector_ci_high"
                ],

                component(
                    vel_mean,
                    0,
                ),

                component(
                    vel_low,
                    0,
                ),

                component(
                    vel_high,
                    0,
                ),

                component(
                    vel_mean,
                    1,
                ),

                component(
                    vel_low,
                    1,
                ),

                component(
                    vel_high,
                    1,
                ),

                component(
                    vel_mean,
                    2,
                ),

                component(
                    vel_low,
                    2,
                ),

                component(
                    vel_high,
                    2,
                ),
            ])

        for (
            controller,
            result,
        ) in sorted(
            overall.items(),
            key=lambda item:
                controller_sort_key(
                    item[0]
                ),
        ):

            pos_mean = result[
                "position_per_dim_mean"
            ]

            pos_low = result[
                "position_per_dim_ci_low"
            ]

            pos_high = result[
                "position_per_dim_ci_high"
            ]

            vel_mean = result[
                "velocity_per_dim_mean"
            ]

            vel_low = result[
                "velocity_per_dim_ci_low"
            ]

            vel_high = result[
                "velocity_per_dim_ci_high"
            ]

            writer.writerow([
                "OVERALL",

                controller_display_name(
                    controller
                ),

                "",
                "",

                result[
                    "num_position_steps"
                ],

                result[
                    "num_velocity_steps"
                ],

                result[
                    "position_xy_mean"
                ],

                result[
                    "position_xy_ci_low"
                ],

                result[
                    "position_xy_ci_high"
                ],

                component(
                    pos_mean,
                    0,
                ),

                component(
                    pos_low,
                    0,
                ),

                component(
                    pos_high,
                    0,
                ),

                component(
                    pos_mean,
                    1,
                ),

                component(
                    pos_low,
                    1,
                ),

                component(
                    pos_high,
                    1,
                ),

                component(
                    pos_mean,
                    2,
                ),

                component(
                    pos_low,
                    2,
                ),

                component(
                    pos_high,
                    2,
                ),

                result[
                    "velocity_vector_mean"
                ],

                result[
                    "velocity_vector_ci_low"
                ],

                result[
                    "velocity_vector_ci_high"
                ],

                component(
                    vel_mean,
                    0,
                ),

                component(
                    vel_low,
                    0,
                ),

                component(
                    vel_high,
                    0,
                ),

                component(
                    vel_mean,
                    1,
                ),

                component(
                    vel_low,
                    1,
                ),

                component(
                    vel_high,
                    1,
                ),

                component(
                    vel_mean,
                    2,
                ),

                component(
                    vel_low,
                    2,
                ),

                component(
                    vel_high,
                    2,
                ),
            ])

    print(
        f"Saved benchmark summary: "
        f"{summary_path}"
    )


def plot_trajectory_error(
    trajectory,
    aggregated,
    metric_key,
    ci_low_key,
    ci_high_key,
    ylabel,
    title,
    filename,
    plots_path,
):

    controllers = sorted(
        (
            controller
            for traj, controller
            in aggregated.keys()
            if traj == trajectory
        ),
        key=controller_sort_key,
    )

    if not controllers:
        return

    values = []
    errors_low = []
    errors_high = []

    for controller in controllers:

        result = aggregated[
            (
                trajectory,
                controller,
            )
        ]

        (
            mean_value,
            error_low,
            error_high,
        ) = error_bar_from_result(
            result,
            metric_key,
            ci_low_key,
            ci_high_key,
        )

        values.append(
            mean_value
        )

        errors_low.append(
            error_low
        )

        errors_high.append(
            error_high
        )

    x = np.arange(
        len(controllers)
    )

    y_errors = np.asarray([
        errors_low,
        errors_high,
    ])

    plt.figure(
        figsize=(
            max(
                7,
                len(controllers) * 1.5,
            ),
            5,
        )
    )

    bars = plt.bar(
        x,
        values,
        yerr=y_errors,
        capsize=5,
        color=[
            controller_color(
                controller
            )
            for controller
            in controllers
        ],
    )

    plt.xticks(
        x,
        [
            controller_display_name(
                controller
            )
            for controller
            in controllers
        ],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        f"{title}\n"
        f"{trajectory_display_name(trajectory)}"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for (
        bar,
        value,
    ) in zip(
        bars,
        values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.4f}",

            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if SAVE_PLOTS:

        trajectory_plot_path = os.path.join(
            plots_path,
            trajectory,
        )

        os.makedirs(
            trajectory_plot_path,
            exist_ok=True,
        )

        save_path = os.path.join(
            trajectory_plot_path,
            filename,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: "
            f"{save_path}"
        )

    if SHOW_PLOTS:
        plt.show()

    plt.close()



def plot_overall_error(
    overall,
    metric_key,
    ci_low_key,
    ci_high_key,
    ylabel,
    title,
    filename,
    plots_path,
):

    controllers = sorted(
        overall.keys(),
        key=controller_sort_key,
    )

    if not controllers:
        return

    values = []
    errors_low = []
    errors_high = []

    for controller in controllers:

        result = overall[
            controller
        ]

        (
            mean_value,
            error_low,
            error_high,
        ) = error_bar_from_result(
            result,
            metric_key,
            ci_low_key,
            ci_high_key,
        )

        values.append(
            mean_value
        )

        errors_low.append(
            error_low
        )

        errors_high.append(
            error_high
        )

    x = np.arange(
        len(controllers)
    )

    y_errors = np.asarray([
        errors_low,
        errors_high,
    ])

    plt.figure(
        figsize=(
            max(
                7,
                len(controllers) * 1.5,
            ),
            5,
        )
    )

    bars = plt.bar(
        x,
        values,
        yerr=y_errors,
        capsize=5,
        color=[
            controller_color(
                controller
            )
            for controller
            in controllers
        ],
    )

    plt.xticks(
        x,
        [
            controller_display_name(
                controller
            )
            for controller
            in controllers
        ],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for (
        bar,
        value,
    ) in zip(
        bars,
        values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.4f}",

            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if SAVE_PLOTS:

        overall_plot_path = os.path.join(
            plots_path,
            "overall",
        )

        os.makedirs(
            overall_plot_path,
            exist_ok=True,
        )

        save_path = os.path.join(
            overall_plot_path,
            filename,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: "
            f"{save_path}"
        )

    if SHOW_PLOTS:
        plt.show()

    plt.close()




def plot_trajectory_comparison(
    trajectory,
    aggregated,
    plots_path,
):
    """
    Plot velocity-integrated XY trajectories.

    Thin transparent lines show the individual physical
    repetitions. Thick lines show the point-wise mean.

    All paths were reconstructed from [vx, vy, omega] using
    DT = 0.02 s and the normalized initial pose (0, 0, 0).
    """

    controllers = sorted(
        (
            controller
            for traj, controller
            in aggregated.keys()
            if traj == trajectory
        ),
        key=controller_sort_key,
    )

    if not controllers:
        return

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    first_controller = (
        controllers[0]
    )

    first_result = aggregated[
        (
            trajectory,
            first_controller,
        )
    ]


    for reference_run in first_result[
        "reference_trajectory_runs"
    ]:

        ax.plot(
            reference_run[:, 0],
            reference_run[:, 1],
            "--",
            linewidth=0.9,
            color="black",
            alpha=0.12,
            zorder=1,
        )


    reference = first_result[
        "reference_trajectory"
    ]

    ax.plot(
        reference[:, 0],
        reference[:, 1],
        "--",
        linewidth=3.0,
        color="black",
        label="Reference Mean",
        zorder=5,
    )


    for controller in controllers:

        result = aggregated[
            (
                trajectory,
                controller,
            )
        ]

        color = controller_color(
            controller
        )

        for actual_run in result[
            "actual_trajectory_runs"
        ]:

            ax.plot(
                actual_run[:, 0],
                actual_run[:, 1],
                linewidth=1.0,
                color=color,
                alpha=0.18,
                zorder=2,
            )

        actual = result[
            "actual_trajectory"
        ]

        ax.plot(
            actual[:, 0],
            actual[:, 1],
            linewidth=2.8,
            color=color,
            label=(
                f"{controller_display_name(controller)} Mean"
            ),
            zorder=4,
        )

    ax.scatter(
        [0.0],
        [0.0],
        s=70,
        marker="o",
        color="black",
        label="Start",
        zorder=10,
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        f"Trajectory Tracking\n"
        f"{trajectory_display_name(trajectory)}"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    if SAVE_PLOTS:

        trajectory_plot_path = os.path.join(
            plots_path,
            trajectory,
        )

        os.makedirs(
            trajectory_plot_path,
            exist_ok=True,
        )

        save_path = os.path.join(
            trajectory_plot_path,
            "trajectory_comparison.png",
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: "
            f"{save_path}"
        )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_trajectory_mean_only(
    trajectory,
    aggregated,
    plots_path,
):
    """
    Plot only the point-wise mean velocity-integrated trajectory
    for the reference and every controller.

    No individual repetitions are shown.
    """

    controllers = sorted(
        (
            controller
            for traj, controller
            in aggregated.keys()
            if traj == trajectory
        ),
        key=controller_sort_key,
    )

    if not controllers:
        return

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    first_controller = controllers[0]

    reference = aggregated[
        (
            trajectory,
            first_controller,
        )
    ][
        "reference_trajectory"
    ]

    ax.plot(
        reference[:, 0],
        reference[:, 1],
        "--",
        linewidth=3.0,
        color="black",
        label="Reference",
        zorder=5,
    )

    for controller in controllers:

        result = aggregated[
            (
                trajectory,
                controller,
            )
        ]

        actual = result[
            "actual_trajectory"
        ]

        ax.plot(
            actual[:, 0],
            actual[:, 1],
            linewidth=2.8,
            color=controller_color(
                controller
            ),
            label=controller_display_name(
                controller
            ),
            zorder=4,
        )

    ax.scatter(
        [0.0],
        [0.0],
        s=70,
        marker="o",
        color="black",
        label="Start",
        zorder=10,
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        f"Trajectory Tracking — Mean\n"
        f"{trajectory_display_name(trajectory)}"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    if SAVE_PLOTS:

        trajectory_plot_path = os.path.join(
            plots_path,
            trajectory,
        )

        os.makedirs(
            trajectory_plot_path,
            exist_ok=True,
        )

        save_path = os.path.join(
            trajectory_plot_path,
            "trajectory_comparison_mean_only.png",
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: "
            f"{save_path}"
        )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_trajectory_median_only(
    trajectory,
    aggregated,
    plots_path,
):
    """
    Plot only the point-wise median velocity-integrated trajectory
    for the reference and every controller.

    The median X and Y coordinates are calculated independently
    at every timestep across the repeated physical runs.
    No individual repetitions are shown.
    """

    controllers = sorted(
        (
            controller
            for traj, controller
            in aggregated.keys()
            if traj == trajectory
        ),
        key=controller_sort_key,
    )

    if not controllers:
        return

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    first_controller = controllers[0]

    reference = aggregated[
        (
            trajectory,
            first_controller,
        )
    ][
        "reference_trajectory_median"
    ]

    ax.plot(
        reference[:, 0],
        reference[:, 1],
        "--",
        linewidth=3.0,
        color="black",
        label="Reference",
        zorder=5,
    )

    for controller in controllers:

        result = aggregated[
            (
                trajectory,
                controller,
            )
        ]

        actual = result[
            "actual_trajectory_median"
        ]

        ax.plot(
            actual[:, 0],
            actual[:, 1],
            linewidth=2.8,
            color=controller_color(
                controller
            ),
            label=controller_display_name(
                controller
            ),
            zorder=4,
        )

    ax.scatter(
        [0.0],
        [0.0],
        s=70,
        marker="o",
        color="black",
        label="Start",
        zorder=10,
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        f"Trajectory Tracking — Median\n"
        f"{trajectory_display_name(trajectory)}"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    if SAVE_PLOTS:

        trajectory_plot_path = os.path.join(
            plots_path,
            trajectory,
        )

        os.makedirs(
            trajectory_plot_path,
            exist_ok=True,
        )

        save_path = os.path.join(
            trajectory_plot_path,
            "trajectory_comparison_median_only.png",
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: "
            f"{save_path}"
        )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_overall_model_comparison(
    overall_nonopt,
    overall_opt,
    metric_key,
    ci_low_key,
    ci_high_key,
    ylabel,
    title,
    filename,
):

    os.makedirs(
        THESIS_PLOTS_PATH,
        exist_ok=True,
    )

    controller_labels = [
        "IK",
        "Hybrid Offset",
        "Hybrid Offset (DR)",
        "Fully Learned",
        "Fully Learned (DR)",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharey=True,
    )

    datasets = [
        (
            axes[0],
            overall_nonopt,
            "Non-Optimized Simulation Model",
        ),
        (
            axes[1],
            overall_opt,
            "Optimized Simulation Model",
        ),
    ]

    for (
        ax,
        overall,
        panel_title,
    ) in datasets:

        available_labels = []
        values = []
        lows = []
        highs = []
        colors = []

        for label in controller_labels:

            result = (
                get_result_by_display_name(
                    overall,
                    label,
                )
            )

            if result is None:
                continue

            (
                mean_value,
                error_low,
                error_high,
            ) = error_bar_from_result(
                result,
                metric_key,
                ci_low_key,
                ci_high_key,
            )

            available_labels.append(
                label
            )

            values.append(
                mean_value
            )

            lows.append(
                error_low
            )

            highs.append(
                error_high
            )

            colors.append(
                CONTROLLER_COLORS[
                    label
                ]
            )

        x = np.arange(
            len(values)
        )

        bars = ax.bar(
            x,
            values,
            yerr=np.asarray([
                lows,
                highs,
            ]),
            capsize=5,
            color=colors,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            available_labels,
            rotation=20,
            ha="right",
        )

        ax.set_title(
            panel_title
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        for (
            bar,
            value,
        ) in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height(),

                f"{value:.4f}",

                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel(
        ylabel
    )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    save_path = os.path.join(
        THESIS_PLOTS_PATH,
        filename,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved thesis plot: "
        f"{save_path}"
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)



def plot_individual_trajectory_model_comparison(
    aggregated_nonopt,
    aggregated_opt,
    metric_key,
    ci_low_key,
    ci_high_key,
    ylabel,
    title,
    filename,
):

    os.makedirs(
        THESIS_PLOTS_PATH,
        exist_ok=True,
    )

    trajectories = (
        ordered_available_trajectories(
            aggregated_nonopt,
            aggregated_opt,
        )
    )

    controller_labels = [
        "IK",
        "Hybrid Offset",
        "Hybrid Offset (DR)",
        "Fully Learned",
        "Fully Learned (DR)",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 6),
        sharey=True,
    )

    datasets = [
        (
            axes[0],
            aggregated_nonopt,
            "Non-Optimized Simulation Model",
        ),
        (
            axes[1],
            aggregated_opt,
            "Optimized Simulation Model",
        ),
    ]

    x = np.arange(
        len(trajectories)
    )

    number_of_controllers = len(
        controller_labels
    )

    total_width = 0.82

    bar_width = (
        total_width
        / number_of_controllers
    )

    for (
        ax,
        aggregated,
        panel_title,
    ) in datasets:

        for (
            controller_index,
            label,
        ) in enumerate(
            controller_labels
        ):

            values = []
            lows = []
            highs = []

            for trajectory in trajectories:

                result = (
                    get_trajectory_result_by_display_name(
                        aggregated,
                        trajectory,
                        label,
                    )
                )

                if result is None:

                    values.append(
                        np.nan
                    )

                    lows.append(
                        0.0
                    )

                    highs.append(
                        0.0
                    )

                    continue

                (
                    mean_value,
                    error_low,
                    error_high,
                ) = error_bar_from_result(
                    result,
                    metric_key,
                    ci_low_key,
                    ci_high_key,
                )

                values.append(
                    mean_value
                )

                lows.append(
                    error_low
                )

                highs.append(
                    error_high
                )

            offset = (
                controller_index
                - (
                    number_of_controllers
                    - 1
                ) / 2
            ) * bar_width

            ax.bar(
                x + offset,
                values,
                bar_width,
                yerr=np.asarray([
                    lows,
                    highs,
                ]),
                capsize=3,
                color=CONTROLLER_COLORS[
                    label
                ],
                label=label,
            )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                trajectory_display_name(
                    trajectory
                )
                for trajectory
                in trajectories
            ],
            rotation=20,
            ha="right",
        )

        ax.set_title(
            panel_title
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

    axes[0].set_ylabel(
        ylabel
    )

    handles, labels = (
        axes[1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.suptitle(
        title,
        y=1.07,
    )

    fig.tight_layout()

    save_path = os.path.join(
        THESIS_PLOTS_PATH,
        filename,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved thesis plot: "
        f"{save_path}"
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_u_curve_difficulty(
    aggregated_nonopt,
    aggregated_opt,
    metric_key,
    ci_low_key,
    ci_high_key,
    ylabel,
    title,
    filename,
):

    os.makedirs(
        THESIS_PLOTS_PATH,
        exist_ok=True,
    )

    available = (
        ordered_available_trajectories(
            aggregated_nonopt,
            aggregated_opt,
        )
    )

    u5 = find_trajectory(
        available,
        [
            "connected_u_curves_5",
            "connected_u_curve_5",
            "u_curves_5",
            "u_curve_5",
        ],
    )

    u8 = find_trajectory(
        available,
        [
            "connected_u_curves_8",
            "connected_u_curve_8",
            "u_curves_8",
            "u_curve_8",
        ],
    )

    if (
        u5 is None
        or u8 is None
    ):

        print()
        print(
            "[WARNING] U-Curve difficulty plot skipped."
        )

        print(
            "Could not find both U5 and U8."
        )

        print(
            f"Available trajectories: "
            f"{available}"
        )

        return

    controller_labels = [
        "IK",
        "Hybrid Offset",
        "Hybrid Offset (DR)",
        "Fully Learned",
        "Fully Learned (DR)",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        sharey=True,
    )

    datasets = [
        (
            axes[0],
            aggregated_nonopt,
            "Non-Optimized Simulation Model",
        ),
        (
            axes[1],
            aggregated_opt,
            "Optimized Simulation Model",
        ),
    ]

    group_x = np.arange(
        len(controller_labels)
    )

    bar_width = 0.32

    all_values = []

    for (
        _,
        aggregated,
        _,
    ) in datasets:

        for controller_label in controller_labels:

            for trajectory in [
                u5,
                u8,
            ]:

                result = (
                    get_trajectory_result_by_display_name(
                        aggregated,
                        trajectory,
                        controller_label,
                    )
                )

                if result is not None:

                    all_values.append(
                        result[
                            metric_key
                        ]
                    )

    if all_values:

        max_value_global = max(
            all_values
        )

    else:

        max_value_global = 1.0

    text_offset = (
        max_value_global
        * 0.045
    )

    line_offset = (
        max_value_global
        * 0.015
    )

    for (
        ax,
        aggregated,
        panel_title,
    ) in datasets:

        panel_max = 0.0

        for (
            index,
            controller_label,
        ) in enumerate(
            controller_labels
        ):

            result_u5 = (
                get_trajectory_result_by_display_name(
                    aggregated,
                    u5,
                    controller_label,
                )
            )

            result_u8 = (
                get_trajectory_result_by_display_name(
                    aggregated,
                    u8,
                    controller_label,
                )
            )

            if (
                result_u5 is None
                or result_u8 is None
            ):
                continue

            (
                value_u5,
                low_u5,
                high_u5,
            ) = error_bar_from_result(
                result_u5,
                metric_key,
                ci_low_key,
                ci_high_key,
            )

            (
                value_u8,
                low_u8,
                high_u8,
            ) = error_bar_from_result(
                result_u8,
                metric_key,
                ci_low_key,
                ci_high_key,
            )

            color = CONTROLLER_COLORS[
                controller_label
            ]

            x_u5 = (
                group_x[index]
                - bar_width / 2
            )

            x_u8 = (
                group_x[index]
                + bar_width / 2
            )

            ax.bar(
                x_u5,
                value_u5,
                width=bar_width,
                yerr=np.asarray(
                    [
                        [low_u5],
                        [high_u5],
                    ]
                ),
                capsize=4,
                color=color,
                edgecolor=color,
                linewidth=1.2,
                zorder=2,
            )


            ax.bar(
                x_u8,
                value_u8,
                width=bar_width,
                yerr=np.asarray(
                    [
                        [low_u8],
                        [high_u8],
                    ]
                ),
                capsize=4,
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                hatch="///",
                zorder=2,
            )


            ax.plot(
                [
                    x_u5,
                    x_u8,
                ],
                [
                    value_u5 + line_offset,
                    value_u8 + line_offset,
                ],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4,
                zorder=3,
            )


            if (
                np.isfinite(value_u5)
                and value_u5 != 0
            ):

                percentage_change = (
                    (
                        value_u8
                        - value_u5
                    )
                    / value_u5
                ) * 100.0

                percentage_text = (
                    f"{percentage_change:+.1f}\\%"
                )

                annotation_y = (
                    max(
                        value_u5 + high_u5,
                        value_u8 + high_u8,
                    )
                    + text_offset
                )

                ax.text(
                    (
                        x_u5
                        + x_u8
                    ) / 2,
                    annotation_y,
                    percentage_text,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )

                panel_max = max(
                    panel_max,
                    annotation_y
                    + text_offset,
                )


            ax.text(
                x_u5,
                value_u5,
                f"{value_u5:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

            ax.text(
                x_u8,
                value_u8,
                f"{value_u8:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

            panel_max = max(
                panel_max,
                value_u5 + high_u5,
                value_u8 + high_u8,
            )


        ax.set_xticks(
            group_x
        )

        ax.set_xticklabels(
            controller_labels,
            rotation=20,
            ha="right",
        )

        ax.set_title(
            panel_title
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
            zorder=0,
        )

        if panel_max > 0:

            ax.set_ylim(
                bottom=0,
                top=panel_max * 1.10,
            )

    axes[0].set_ylabel(
        ylabel
    )


    trajectory_legend = [
        Patch(
            facecolor="gray",
            edgecolor="gray",
            label="Connected U-Curves (5)",
        ),

        Patch(
            facecolor="white",
            edgecolor="gray",
            hatch="///",
            label="Connected U-Curves (8)",
        ),
    ]

    fig.legend(
        handles=trajectory_legend,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.suptitle(
        title,
        y=1.08,
    )

    fig.tight_layout()

    save_path = os.path.join(
        THESIS_PLOTS_PATH,
        filename,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved thesis plot: "
        f"{save_path}"
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_representative_trajectories(
    aggregated_opt,
):
    """
    Plot representative optimized-model physical trajectories.

    Trajectories are reconstructed from body-frame velocities
    using DT = 0.02 s and initial pose (0, 0, 0). Individual
    repetitions are shown faintly and the point-wise mean is
    drawn on top.
    """

    available = (
        ordered_available_trajectories(
            aggregated_opt
        )
    )

    changing = find_trajectory(
        available,
        [
            "changing_velocities",
            "changing_velocity",
        ],
    )

    u8 = find_trajectory(
        available,
        [
            "connected_u_curves_8",
            "connected_u_curve_8",
            "u_curves_8",
            "u_curve_8",
        ],
    )

    if (
        changing is None
        or u8 is None
    ):

        print()
        print(
            "[WARNING] Representative "
            "trajectory figure skipped."
        )

        print(
            f"Available trajectories: "
            f"{available}"
        )

        return

    selected = [
        changing,
        u8,
    ]

    panel_titles = [
        "Changing Velocities",
        "Connected U-Curves (8)",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    for (
        ax,
        trajectory,
        panel_title,
    ) in zip(
        axes,
        selected,
        panel_titles,
    ):

        controllers = sorted(
            [
                controller
                for (
                    traj,
                    controller,
                )
                in aggregated_opt.keys()
                if traj == trajectory
            ],
            key=controller_sort_key,
        )

        if not controllers:
            continue

        first_result = aggregated_opt[
            (
                trajectory,
                controllers[0],
            )
        ]

        # Individual reference repetitions.
        for reference_run in first_result[
            "reference_trajectory_runs"
        ]:

            ax.plot(
                reference_run[:, 0],
                reference_run[:, 1],
                "--",
                linewidth=0.8,
                color="black",
                alpha=0.10,
                zorder=1,
            )

        # Mean reference.
        reference = first_result[
            "reference_trajectory"
        ]

        ax.plot(
            reference[:, 0],
            reference[:, 1],
            "--",
            linewidth=2.5,
            color="black",
            label="Reference Mean",
            zorder=5,
        )

        for controller in controllers:

            result = aggregated_opt[
                (
                    trajectory,
                    controller,
                )
            ]

            color = controller_color(
                controller
            )

            # Individual repetitions.
            for actual_run in result[
                "actual_trajectory_runs"
            ]:

                ax.plot(
                    actual_run[:, 0],
                    actual_run[:, 1],
                    linewidth=0.9,
                    color=color,
                    alpha=0.18,
                    zorder=2,
                )

            # Mean trajectory.
            actual = result[
                "actual_trajectory"
            ]

            ax.plot(
                actual[:, 0],
                actual[:, 1],
                linewidth=2.6,
                color=color,
                label=(
                    f"{controller_display_name(controller)} Mean"
                ),
                zorder=4,
            )

        ax.scatter(
            [0.0],
            [0.0],
            color="black",
            marker="o",
            s=45,
            zorder=10,
            label="Start",
        )

        ax.set_xlabel(
            "X [m]"
        )

        ax.set_ylabel(
            "Y [m]"
        )

        ax.set_title(
            panel_title
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    handles, labels = (
        axes[1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.suptitle(
        "Physical Robot Trajectory Tracking",
        y=1.06,
    )

    fig.tight_layout()

    save_path = os.path.join(
        THESIS_PLOTS_PATH,
        "representative_trajectory_tracking.png",
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved thesis plot: "
        f"{save_path}"
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def plot_representative_trajectories_summary(
    aggregated_opt,
    statistic="mean",
):
    """
    Plot the two representative optimized-model physical
    trajectories using only one summary trajectory per controller.

    statistic:
        "mean"   -> point-wise arithmetic mean
        "median" -> point-wise median
    """

    if statistic == "mean":
        reference_key = "reference_trajectory"
        actual_key = "actual_trajectory"
        suffix = "mean_only"
        statistic_title = "Mean Trajectories"

    elif statistic == "median":
        reference_key = "reference_trajectory_median"
        actual_key = "actual_trajectory_median"
        suffix = "median_only"
        statistic_title = "Median Trajectories"

    else:
        raise ValueError(
            "statistic must be 'mean' or 'median'"
        )

    available = (
        ordered_available_trajectories(
            aggregated_opt
        )
    )

    changing = find_trajectory(
        available,
        [
            "changing_velocities",
            "changing_velocity",
        ],
    )

    u8 = find_trajectory(
        available,
        [
            "connected_u_curves_8",
            "connected_u_curve_8",
            "u_curves_8",
            "u_curve_8",
        ],
    )

    if (
        changing is None
        or u8 is None
    ):
        print()
        print(
            f"[WARNING] Representative {statistic} "
            "trajectory figure skipped."
        )
        print(
            f"Available trajectories: "
            f"{available}"
        )
        return

    selected = [
        changing,
        u8,
    ]

    panel_titles = [
        "Changing Velocities",
        "Connected U-Curves (8)",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    for (
        ax,
        trajectory,
        panel_title,
    ) in zip(
        axes,
        selected,
        panel_titles,
    ):

        controllers = sorted(
            [
                controller
                for (
                    traj,
                    controller,
                )
                in aggregated_opt.keys()
                if traj == trajectory
            ],
            key=controller_sort_key,
        )

        if not controllers:
            continue

        first_result = aggregated_opt[
            (
                trajectory,
                controllers[0],
            )
        ]

        reference = first_result[
            reference_key
        ]

        ax.plot(
            reference[:, 0],
            reference[:, 1],
            "--",
            linewidth=2.5,
            color="black",
            label="Reference",
            zorder=5,
        )

        for controller in controllers:

            result = aggregated_opt[
                (
                    trajectory,
                    controller,
                )
            ]

            actual = result[
                actual_key
            ]

            ax.plot(
                actual[:, 0],
                actual[:, 1],
                linewidth=2.6,
                color=controller_color(
                    controller
                ),
                label=controller_display_name(
                    controller
                ),
                zorder=4,
            )

        ax.scatter(
            [0.0],
            [0.0],
            color="black",
            marker="o",
            s=45,
            zorder=10,
            label="Start",
        )

        ax.set_xlabel(
            "X [m]"
        )

        ax.set_ylabel(
            "Y [m]"
        )

        ax.set_title(
            panel_title
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    handles, labels = (
        axes[1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.suptitle(
        f"Physical Robot Trajectory Tracking — "
        f"{statistic_title}",
        y=1.06,
    )

    fig.tight_layout()

    save_path = os.path.join(
        THESIS_PLOTS_PATH,
        f"representative_trajectory_tracking_{suffix}.png",
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved thesis plot: "
        f"{save_path}"
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def process_dataset(
    dataset_name,
    base_path,
):

    print()
    print(
        "=" * 80
    )
    print(
        dataset_name
    )
    print(
        "=" * 80
    )

    print(
        f"Searching benchmark runs in: "
        f"{base_path}"
    )

    runs = discover_runs(
        base_path
    )

    if not runs:

        raise RuntimeError(
            f"No valid benchmark runs found "
            f"in {base_path}"
        )

    print(
        f"Found {len(runs)} "
        f"valid recorded runs."
    )

    print_controller_grouping(
        runs
    )

    aggregated = aggregate_results(
        runs
    )

    overall = compute_overall_results(
        aggregated
    )

    print_results(
        aggregated,
        overall,
        dataset_name,
    )

    summary_path = os.path.join(
        base_path,
        "benchmark_summary.csv",
    )

    save_summary_csv(
        aggregated,
        overall,
        summary_path,
    )

    plots_path = os.path.join(
        base_path,
        "plots",
    )

    os.makedirs(
        plots_path,
        exist_ok=True,
    )

    metric_name = METRIC.upper()

    trajectories = (
        ordered_available_trajectories(
            aggregated
        )
    )


    for trajectory in trajectories:

        print()

        print(
            f"Generating plots for: "
            f"{trajectory_display_name(trajectory)}"
        )

        plot_trajectory_error(
            trajectory=trajectory,
            aggregated=aggregated,

            metric_key=(
                "position_xy_mean"
            ),

            ci_low_key=(
                "position_xy_ci_low"
            ),

            ci_high_key=(
                "position_xy_ci_high"
            ),

            ylabel=metric_name,

            title=(
                f"Position Tracking Error | "
                f"{metric_name}"
            ),

            filename=(
                f"position_xy_"
                f"{METRIC}.png"
            ),

            plots_path=plots_path,
        )

        plot_trajectory_error(
            trajectory=trajectory,
            aggregated=aggregated,

            metric_key=(
                "velocity_vector_mean"
            ),

            ci_low_key=(
                "velocity_vector_ci_low"
            ),

            ci_high_key=(
                "velocity_vector_ci_high"
            ),

            ylabel=metric_name,

            title=(
                f"Velocity Tracking Error | "
                f"{metric_name}"
            ),

            filename=(
                f"velocity_vector_"
                f"{METRIC}.png"
            ),

            plots_path=plots_path,
        )

        plot_trajectory_comparison(
            trajectory=trajectory,
            aggregated=aggregated,
            plots_path=plots_path,
        )

        # Mean-only trajectory figure.
        plot_trajectory_mean_only(
            trajectory=trajectory,
            aggregated=aggregated,
            plots_path=plots_path,
        )

        # Median-only trajectory figure.
        plot_trajectory_median_only(
            trajectory=trajectory,
            aggregated=aggregated,
            plots_path=plots_path,
        )


    plot_overall_error(
        overall=overall,

        metric_key=(
            "position_xy_mean"
        ),

        ci_low_key=(
            "position_xy_ci_low"
        ),

        ci_high_key=(
            "position_xy_ci_high"
        ),

        ylabel=metric_name,

        title=(
            f"Overall Position Tracking Error | "
            f"{metric_name}"
        ),

        filename=(
            f"position_xy_{METRIC}.png"
        ),

        plots_path=plots_path,
    )

    plot_overall_error(
        overall=overall,

        metric_key=(
            "velocity_vector_mean"
        ),

        ci_low_key=(
            "velocity_vector_ci_low"
        ),

        ci_high_key=(
            "velocity_vector_ci_high"
        ),

        ylabel=metric_name,

        title=(
            f"Overall Velocity Tracking Error | "
            f"{metric_name}"
        ),

        filename=(
            f"velocity_vector_"
            f"{METRIC}.png"
        ),

        plots_path=plots_path,
    )

    return (
        runs,
        aggregated,
        overall,
    )



def main():

    os.makedirs(
        THESIS_PLOTS_PATH,
        exist_ok=True,
    )

    (
        runs_nonopt,
        aggregated_nonopt,
        overall_nonopt,
    ) = process_dataset(
        dataset_name=(
            "POLICIES TRAINED WITH "
            "NON-OPTIMIZED SIMULATION MODEL"
        ),
        base_path=(
            NON_OPTIMIZED_BASE_PATH
        ),
    )


    (
        runs_opt,
        aggregated_opt,
        overall_opt,
    ) = process_dataset(
        dataset_name=(
            "POLICIES TRAINED WITH "
            "OPTIMIZED SIMULATION MODEL"
        ),
        base_path=(
            OPTIMIZED_BASE_PATH
        ),
    )

    metric_name = METRIC.upper()

    print()
    print(
        "=" * 80
    )
    print(
        "GENERATING THESIS COMPARISON PLOTS"
    )
    print(
        "=" * 80
    )


    plot_overall_model_comparison(
        overall_nonopt=overall_nonopt,
        overall_opt=overall_opt,

        metric_key=(
            "position_xy_mean"
        ),

        ci_low_key=(
            "position_xy_ci_low"
        ),

        ci_high_key=(
            "position_xy_ci_high"
        ),

        ylabel=metric_name,

        title=(
            "Physical Robot — "
            "Overall Position Tracking"
        ),

        filename=(
            "overall_position_"
            "model_comparison.png"
        ),
    )

    plot_overall_model_comparison(
        overall_nonopt=overall_nonopt,
        overall_opt=overall_opt,

        metric_key=(
            "velocity_vector_mean"
        ),

        ci_low_key=(
            "velocity_vector_ci_low"
        ),

        ci_high_key=(
            "velocity_vector_ci_high"
        ),

        ylabel=metric_name,

        title=(
            "Physical Robot — "
            "Overall Velocity Tracking"
        ),

        filename=(
            "overall_velocity_"
            "model_comparison.png"
        ),
    )


    plot_individual_trajectory_model_comparison(
        aggregated_nonopt=aggregated_nonopt,
        aggregated_opt=aggregated_opt,

        metric_key=(
            "position_xy_mean"
        ),

        ci_low_key=(
            "position_xy_ci_low"
        ),

        ci_high_key=(
            "position_xy_ci_high"
        ),

        ylabel=metric_name,

        title=(
            "Physical Robot — "
            "Position Tracking by Trajectory"
        ),

        filename=(
            "individual_trajectory_position_"
            "model_comparison.png"
        ),
    )

    plot_individual_trajectory_model_comparison(
        aggregated_nonopt=aggregated_nonopt,
        aggregated_opt=aggregated_opt,

        metric_key=(
            "velocity_vector_mean"
        ),

        ci_low_key=(
            "velocity_vector_ci_low"
        ),

        ci_high_key=(
            "velocity_vector_ci_high"
        ),

        ylabel=metric_name,

        title=(
            "Physical Robot — "
            "Velocity Tracking by Trajectory"
        ),

        filename=(
            "individual_trajectory_velocity_"
            "model_comparison.png"
        ),
    )


    plot_u_curve_difficulty(
        aggregated_nonopt=aggregated_nonopt,
        aggregated_opt=aggregated_opt,

        metric_key=(
            "position_xy_mean"
        ),

        ci_low_key=(
            "position_xy_ci_low"
        ),

        ci_high_key=(
            "position_xy_ci_high"
        ),

        ylabel=metric_name,

        title=(
            "Physical Robot — "
            "Position Tracking: "
            "Connected U-Curves (5) vs (8)"
        ),

        filename=(
            "u_curve_position_difficulty.png"
        ),
    )

    plot_u_curve_difficulty(
        aggregated_nonopt=aggregated_nonopt,
        aggregated_opt=aggregated_opt,

        metric_key=(
            "velocity_vector_mean"
        ),

        ci_low_key=(
            "velocity_vector_ci_low"
        ),

        ci_high_key=(
            "velocity_vector_ci_high"
        ),

        ylabel=metric_name,

        title=(
            "Physical Robot — "
            "Velocity Tracking: "
            "Connected U-Curves (5) vs (8)"
        ),

        filename=(
            "u_curve_velocity_difficulty.png"
        ),
    )


    plot_representative_trajectories(
        aggregated_opt
    )

    # Mean-only representative thesis figure.
    plot_representative_trajectories_summary(
        aggregated_opt,
        statistic="mean",
    )

    # Median-only representative thesis figure.
    plot_representative_trajectories_summary(
        aggregated_opt,
        statistic="median",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "BENCHMARK COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        f"Non-optimized plots: "
        f"{os.path.join(NON_OPTIMIZED_BASE_PATH, 'plots')}"
    )

    print(
        f"Optimized plots: "
        f"{os.path.join(OPTIMIZED_BASE_PATH, 'plots')}"
    )

    print(
        f"Thesis comparison plots: "
        f"{THESIS_PLOTS_PATH}"
    )


if __name__ == "__main__":
    main()