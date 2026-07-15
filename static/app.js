document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const sourcesPanel = document.getElementById('sourcesPanel');
    const sourcesList = document.getElementById('sourcesList');
    const toggleSourcesBtn = document.getElementById('toggleSourcesBtn');
    const closeSourcesBtn = document.getElementById('closeSourcesBtn');
    const systemTime = document.getElementById('systemTime');

    // Set initial system time
    if (systemTime) {
        systemTime.textContent = formatTime(new Date());
    }

    // Toggle sources panel visibility
    toggleSourcesBtn.addEventListener('click', () => {
        sourcesPanel.classList.toggle('collapsed');
    });

    closeSourcesBtn.addEventListener('click', () => {
        sourcesPanel.classList.add('collapsed');
    });

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        // Clear input and add user message
        userInput.value = '';
        addMessage(query, 'user');
        
        // Auto open sources panel on mobile/desktop if closed
        sourcesPanel.classList.remove('collapsed');

        // Add typing indicator
        const typingId = addTypingIndicator();
        scrollToBottom();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query, k: 3 })
            });

            // Remove typing indicator
            removeElement(typingId);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch answer.');
            }

            const data = await response.json();
            
            // Add assistant response
            addMessage(data.answer, 'assistant');
            
            // Render retrieved sources
            renderSources(data.sources || data.chunks);

        } catch (error) {
            console.error('Error fetching RAG query:', error);
            // Remove typing indicator in case it's still there
            removeElement(typingId);
            addMessage(`Sorry, an error occurred: ${error.message}`, 'assistant error-message');
        }

        scrollToBottom();
    });

    // Helper: format date to HH:MM AM/PM
    function formatTime(date) {
        let hours = date.getHours();
        let minutes = date.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12; // 12 instead of 0
        minutes = minutes < 10 ? '0' + minutes : minutes;
        return `${hours}:${minutes} ${ampm}`;
    }

    // Helper: append message to chat area
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = `avatar ${sender}-avatar`;
        avatarDiv.textContent = sender.startsWith('user') ? 'U' : 'AI';

        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'message-bubble-wrapper';

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        const p = document.createElement('p');
        p.textContent = text;
        bubbleDiv.appendChild(p);

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = formatTime(new Date());

        wrapperDiv.appendChild(bubbleDiv);
        wrapperDiv.appendChild(timeSpan);

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(wrapperDiv);

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Helper: add typing bubble
    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.id = id;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar assistant-avatar';
        avatarDiv.textContent = 'AI';

        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'message-bubble-wrapper';

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';

        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            indicatorDiv.appendChild(dot);
        }

        bubbleDiv.appendChild(indicatorDiv);
        wrapperDiv.appendChild(bubbleDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(wrapperDiv);

        chatMessages.appendChild(messageDiv);
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Render retrieved sources list in side panel
    function renderSources(sources) {
        if (!sources || sources.length === 0) {
            sourcesList.innerHTML = `
                <div class="empty-sources-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 9V12M12 12V15M12 12H9M12 12H15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <p>No sources found</p>
                    <span>No document chunks were retrieved for the last query.</span>
                </div>
            `;
            return;
        }

        sourcesList.innerHTML = '';
        sources.forEach((src, index) => {
            const card = document.createElement('div');
            card.className = 'source-card';

            const score = (src.score !== undefined && src.score !== null) ? parseFloat(src.score).toFixed(3) : 'N/A';

            if (src.source === 'web') {
                const title = src.title || 'Web Result';
                const url = src.url || '#';
                const content = src.content || '';
                card.innerHTML = `
                    <div class="source-meta">
                        <span class="source-rank">Source #${index + 1}</span>
                        <span class="source-badge source-badge-web">🌐 WEB</span>
                        <span class="source-score">Score: ${score}</span>
                    </div>
                    <div class="source-doc-info">
                        <a href="${escapeHtml(url)}" target="_blank" class="source-link">${escapeHtml(title)}</a>
                    </div>
                    <div class="source-text">"${escapeHtml(content.substring(0, 300))}${content.length > 300 ? '...' : ''}"</div>
                `;
            } else {
                const docText = src.document || '';
                card.innerHTML = `
                    <div class="source-meta">
                        <span class="source-rank">Source #${index + 1}</span>
                        <span class="source-badge source-badge-db">🗄️ DB</span>
                        <span class="source-score">Score: ${score}</span>
                    </div>
                    <div class="source-doc-info">Doc ID: ${src.doc_id} &bull; Chunk: ${src.chunk_id}</div>
                    <div class="source-text">"${escapeHtml(docText.substring(0, 300))}${docText.length > 300 ? '...' : ''}"</div>
                `;
            }
            sourcesList.appendChild(card);
        });
    }

    function escapeHtml(text) {
        if (text === undefined || text === null) {
            return '';
        }
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, function(m) { return map[m]; });
    }
});
