// Shared JavaScript for Both Pages

// Index Page Specific: Form Validation
if (document.querySelector('form[action="/predict"]')) {
  document.querySelector('form').addEventListener('submit', function(e) {
    const inputs = this.querySelectorAll('input[type="number"]');
    let valid = true;
    inputs.forEach(input => {
      if (input.value === '' || isNaN(input.value)) {
        valid = false;
        input.classList.add('is-invalid');
      } else {
        input.classList.remove('is-invalid');
      }
    });
    if (!valid) {
      e.preventDefault();
      const errorDiv = document.getElementById('error-message');
      errorDiv.textContent = 'Please fill in all fields with valid numbers.';
      errorDiv.classList.remove('d-none');
    }
  });

  // Initialize Bootstrap Tooltips for info icons
  var tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
}

// Result Page Specific: Dynamic Prediction Simulation
if (document.getElementById('prediction-text')) {
  document.addEventListener('DOMContentLoaded', function() {
    const outcomes = [
      { text: 'Chase Successful', prob: 85.50, class: 'prediction-success' },
      { text: 'Chase Failed', prob: 42.30, class: 'prediction-fail' },
      { text: 'Too Close to Call', prob: 50.00, class: 'prediction-success' }
    ];
    const randomOutcome = outcomes[Math.floor(Math.random() * outcomes.length)];
    
    const predText = document.getElementById('prediction-text');
    predText.textContent = randomOutcome.text;
    predText.className = randomOutcome.class;
    
    const probText = document.getElementById('probability-text');
    probText.textContent = randomOutcome.prob.toFixed(2) + '%';
    
    // Add fade-in animation
    const resultDiv = document.querySelector('.glass');
    resultDiv.style.opacity = '0';
    resultDiv.style.transition = 'opacity 0.5s';
    setTimeout(() => {
      resultDiv.style.opacity = '1';
    }, 100);
  });
}