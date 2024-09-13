document.getElementById('prediction-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify(Object.fromEntries(formData)),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction-rf').classList.add('alert');
        document.getElementById('prediction-rf').classList.add(data.rf_prediction === 'malignant' ? 'alert-danger' : 'alert-success');
        document.getElementById('prediction-rf').textContent = `RF Prediction: ${data.rf_prediction}`;
    })
    .catch(error => console.error('Error:', error));
});
