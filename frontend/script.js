document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const aiScoreInput = document.getElementById('AIScore');
    const rangeValue = document.getElementById('rangeValue');
    const resultContainer = document.getElementById('resultContainer');
    const predictionResult = document.getElementById('predictionResult');
    const resultMsg = document.getElementById('resultMsg');
    const submitBtn = document.getElementById('submitBtn');
    const resetBtn = document.getElementById('resetBtn');

    // Update range value display
    aiScoreInput.addEventListener('input', (e) => {
        rangeValue.textContent = e.target.value;
    });

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Loading state
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Convert numbers
            if (key === 'Experience (Years)' || key === 'Salary Expectation ($)' || key === 'Projects Count' || key === 'AI Score (0-100)') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        });

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (result.status === 'success') {
                showResult(result.prediction, data.Name);
            } else {
                alert('Error: ' + result.message);
            }
        } catch (error) {
            console.error('Error fetching prediction:', error);
            alert('Could not connect to the backend server. Make sure app.py is running.');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });

    function showResult(prediction, name) {
        form.classList.add('hidden');
        resultContainer.classList.remove('hidden');

        predictionResult.textContent = prediction;
        predictionResult.className = 'prediction-value ' + prediction.toLowerCase();

        if (prediction.toLowerCase() === 'hire') {
            resultMsg.textContent = `${name} matches our ideal candidate profile perfectly. Proceeding with the offer!`;
        } else {
            resultMsg.textContent = `${name} doesn't quite meet the requirements for this role yet. We'll keep them in our talent pool.`;
        }
    }

    resetBtn.addEventListener('click', () => {
        form.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        form.reset();
        rangeValue.textContent = '50';
    });
});
