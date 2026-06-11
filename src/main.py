"""
Run the full animated linear regression pipeline:

    load data → train model → plot loss / snapshots / weight path → animate
"""

from src.data import load_and_prepare_data
from src.model import LinearRegressionGD
from src.visualize import OUTPUT_DIR, animate_regression, plot_all


def main(
    lr: float = 0.01,
    n_epochs: int = 200,
    show_plots: bool = True,
    save_animation: bool = True,
) -> None:
    # --- Data ---
    dataset = load_and_prepare_data()
    X_train, y_train = dataset.X_train, dataset.y_train

    # --- Train ---
    model = LinearRegressionGD(lr=lr, n_epochs=n_epochs)
    model.fit(X_train, y_train)

    print(f"Initial loss: {model.history['loss'][0]:.4f}")
    print(f"Final loss:   {model.history['loss'][-1]:.4f}")
    print(f"Final w={model.w:.4f}, b={model.b:.4f}")

    # --- Static plots (saved to outputs/) ---
    plot_all(X_train, y_train, model.history, show=show_plots)

    # --- Animated regression line ---
    anim_path = OUTPUT_DIR / "regression_animation.gif" if save_animation else None
    animate_regression(
        X_train,
        y_train,
        model.history,
        interval=50,
        frame_step=max(1, n_epochs // 100),  # cap ~100 frames
        save_path=anim_path,
        show=show_plots,
    )

    if save_animation:
        print(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
