document.addEventListener('DOMContentLoaded', () => {
    // Fetch past predictions and performance metrics
    fetch('/api/past_predictions')
        .then(response => response.json())
        .then(data => {
            populatePredictionsTable(data.past_predictions);
            renderMetricsChart(data.metrics);
            renderFeatureDistributionChart(data.feature_distribution);
        })
        .catch(error => console.error('Error fetching data:', error));
});

function populatePredictionsTable(predictions) {
    const tableBody = document.getElementById('predictions-table-body');
    tableBody.innerHTML = predictions.map(prediction => `
        <tr>
            <td>${prediction.id}</td>
            <td>${prediction.radius}</td>
            <td>${prediction.texture}</td>
            <td>${prediction.perimeter}</td>
            <td>${prediction.area}</td>
            <td>${prediction.prediction}</td>
            <td>${prediction.date}</td>
        </tr>
    `).join('');
}

function renderMetricsChart(metrics) {
    const ctx = document.getElementById('metrics-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Performance Metrics',
                data: [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function renderFeatureDistributionChart(distribution) {
    const ctx = document.getElementById('feature-distribution-chart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: distribution.labels,
            datasets: [{
                label: 'Feature Distribution',
                data: distribution.values,
                backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)', 'rgba(75, 192, 192, 0.2)'],
                borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)'],
                borderWidth: 1
            }]
        }
    });
}
