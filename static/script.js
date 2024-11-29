// Function to handle YouTube link submission
function submitYoutubeLink() {
    const youtubeLink = document.getElementById('youtube-link').value;
    
    if (!youtubeLink) {
        alert('Please enter a YouTube link');
        return;
    }

    fetch('/submit_youtube_link', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `youtube_link=${youtubeLink}`
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('link-status').textContent = data.message;
        document.getElementById('send-button').disabled = false;
        document.getElementById('question-input').disabled = false;
    })
    .catch(error => console.error('Error:', error));
}

// Function to handle question submission and display chat
function askQuestion() {
    const question = document.getElementById('question-input').value;

    if (!question) {
        alert('Please ask a question');
        return;
    }

    const chatBox = document.getElementById('chat-box');

    // Display the question in chat
    const questionElement = document.createElement('div');
    questionElement.className = 'chat-message question';
    questionElement.textContent = question;
    chatBox.appendChild(questionElement);

    // Clear the input
    document.getElementById('question-input').value = '';

    // Get the model response
    fetch('/ask_question', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `question=${question}`
    })
    .then(response => response.json())
    .then(data => {
        // Display the model's answer in chat
        const answerElement = document.createElement('div');
        answerElement.className = 'chat-message answer';
        answerElement.textContent = data.answer;
        chatBox.appendChild(answerElement);

        // Scroll to the bottom of the chat
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
}

// Function to clear chat and reset inputs
function clearChat() {
    // Clear chat messages
    document.getElementById('chat-box').innerHTML = '';
    
    // Clear the YouTube link input
    document.getElementById('youtube-link').value = '';
    
    // Clear the link status message
    document.getElementById('link-status').textContent = '';

    // Disable the question input and Send button
    document.getElementById('question-input').disabled = true;
    document.getElementById('send-button').disabled = true;
}