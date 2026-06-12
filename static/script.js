// ===================== Helpers =====================

// Lightweight toast notifications (replaces blocking alert())
function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    if (!toast) {
        alert(message);
        return;
    }
    toast.textContent = message;
    toast.className = 'toast show' + (isError ? ' error' : '');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => {
        toast.className = 'toast' + (isError ? ' error' : '');
    }, 3500);
}

function scrollChatToBottom() {
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
}

// ===================== Chatbot page =====================

function setChatLoading(isLoading) {
    const sendBtn = document.getElementById('send-button');
    const input = document.getElementById('question-input');
    if (!sendBtn || !input) return;
    sendBtn.disabled = isLoading;
    input.disabled = isLoading;
    sendBtn.innerHTML = isLoading
        ? '<span class="spinner"></span>Thinking'
        : 'Send';
}

function showTypingIndicator() {
    const chatBox = document.getElementById('chat-box');
    const el = document.createElement('div');
    el.className = 'chat-message answer';
    el.id = 'typing-indicator';
    el.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    chatBox.appendChild(el);
    scrollChatToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// Function to handle question submission and display chat
function askQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();

    if (!question) {
        showToast('Please type a question first.', true);
        return;
    }

    const chatBox = document.getElementById('chat-box');

    // Remove the empty-state placeholder on first message
    const emptyState = chatBox.querySelector('.chat-empty');
    if (emptyState) emptyState.remove();

    // Display the question in chat
    const questionElement = document.createElement('div');
    questionElement.className = 'chat-message question';
    questionElement.textContent = question;
    chatBox.appendChild(questionElement);

    // Clear the input and show loading state
    input.value = '';
    setChatLoading(true);
    showTypingIndicator();
    scrollChatToBottom();

    // Get the model response
    fetch('/ask_question', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `question=${encodeURIComponent(question)}`
    })
    .then(response => {
        if (!response.ok) throw new Error('Server responded with ' + response.status);
        return response.json();
    })
    .then(data => {
        removeTypingIndicator();
        const answerElement = document.createElement('div');
        answerElement.className = 'chat-message answer';
        answerElement.textContent = data.answer;
        chatBox.appendChild(answerElement);
        scrollChatToBottom();
    })
    .catch(error => {
        console.error('Error:', error);
        removeTypingIndicator();
        const errorElement = document.createElement('div');
        errorElement.className = 'chat-message error';
        errorElement.textContent = '⚠️ Sorry, something went wrong. Please try again.';
        chatBox.appendChild(errorElement);
        scrollChatToBottom();
    })
    .finally(() => {
        setChatLoading(false);
        input.focus();
    });
}

// Function to clear chat
function clearChat() {
    const chatBox = document.getElementById('chat-box');
    if (!chatBox) return;
    chatBox.innerHTML = '<p class="chat-empty">Ask a question about your media to get started.</p>';
}

// ===================== Landing page =====================

// Store selected media
const mediaItems = [];

// Add YouTube Link
function addYoutubeLink() {
    const linkInput = document.getElementById('youtube-link');
    const link = linkInput.value.trim();
    if (!link) {
        showToast('Please enter a YouTube link.', true);
        return;
    }
    if (!/youtu\.?be/i.test(link)) {
        showToast('That doesn’t look like a YouTube link.', true);
        return;
    }
    mediaItems.push({ type: 'YouTube', value: link });
    updateMediaList();
    linkInput.value = '';
    showToast('YouTube link added.');
}

// Add Uploaded Documents
function addDocuments() {
    const fileInput = document.getElementById('upload-docs');
    const files = fileInput.files;
    if (!files || files.length === 0) {
        showToast('Please choose at least one file.', true);
        return;
    }
    for (let i = 0; i < files.length; i++) {
        mediaItems.push({ type: 'Document', value: files[i] });
    }
    updateMediaList();
    fileInput.value = '';
    showToast(`${files.length} file(s) added.`);
}

// Remove a media item by index
function removeMediaItem(index) {
    mediaItems.splice(index, 1);
    updateMediaList();
}

// Update the Media List Display
function updateMediaList() {
    const list = document.getElementById('media-items');
    const empty = document.getElementById('media-empty');
    const insightsBtn = document.getElementById('insights-btn');
    if (!list) return;

    list.innerHTML = '';
    mediaItems.forEach((item, index) => {
        const li = document.createElement('li');

        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = item.type === 'Document' ? 'Doc' : 'Video';

        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = item.type === 'Document' ? item.value.name : item.value;

        const remove = document.createElement('button');
        remove.className = 'remove';
        remove.innerHTML = '&times;';
        remove.title = 'Remove';
        remove.onclick = () => removeMediaItem(index);

        li.appendChild(tag);
        li.appendChild(name);
        li.appendChild(remove);
        list.appendChild(li);
    });

    if (empty) empty.style.display = mediaItems.length ? 'none' : 'block';
    if (insightsBtn) insightsBtn.disabled = mediaItems.length === 0;
}

// Submit and process all media
function submitMedia() {
    const formData = new FormData();
    mediaItems.forEach(item => {
        if (item.type === 'YouTube') {
            formData.append('youtube_links[]', item.value);
        } else if (item.type === 'Document') {
            formData.append('documents[]', item.value);
        }
    });

    return fetch('/submit_media', {
        method: 'POST',
        body: formData,
    });
}

// Navigate to Chatbot Page
async function goToChatbot() {
    if (mediaItems.length === 0) {
        showToast('Please add at least one media item.', true);
        return;
    }

    const overlay = document.getElementById('processing-overlay');
    const insightsBtn = document.getElementById('insights-btn');
    if (overlay) overlay.classList.add('active');
    if (insightsBtn) insightsBtn.disabled = true;

    try {
        const response = await submitMedia();
        if (response.ok) {
            window.location.href = '/chatbot';
        } else {
            const errorData = await response.json().catch(() => ({}));
            if (overlay) overlay.classList.remove('active');
            if (insightsBtn) insightsBtn.disabled = false;
            showToast(`Error: ${errorData.error || 'Failed to submit media.'}`, true);
        }
    } catch (error) {
        console.error('Error submitting media:', error);
        if (overlay) overlay.classList.remove('active');
        if (insightsBtn) insightsBtn.disabled = false;
        showToast('An error occurred while submitting media. Please try again.', true);
    }
}

// ===================== Keyboard shortcuts =====================
document.addEventListener('DOMContentLoaded', () => {
    const questionInput = document.getElementById('question-input');
    if (questionInput) {
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); askQuestion(); }
        });
        questionInput.focus();
    }

    const youtubeInput = document.getElementById('youtube-link');
    if (youtubeInput) {
        youtubeInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); addYoutubeLink(); }
        });
    }
});
