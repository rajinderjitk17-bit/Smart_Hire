document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-range-input]").forEach((input) => {
        const output = document.querySelector(`[data-range-output="${input.dataset.rangeInput}"]`);
        if (!output) {
            return;
        }
        const update = () => {
            output.textContent = input.value;
        };
        input.addEventListener("input", update);
        update();
    });

    document.querySelectorAll(".alert").forEach((alert) => {
        window.setTimeout(() => {
            if (window.bootstrap) {
                const instance = bootstrap.Alert.getOrCreateInstance(alert);
                instance.close();
            }
        }, 5200);
    });
});
