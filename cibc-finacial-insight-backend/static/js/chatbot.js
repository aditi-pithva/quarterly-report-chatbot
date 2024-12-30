document.getElementById("send-button").addEventListener("click", sendMessage);

function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();
    if (message === "") return;

    addMessageToChat("user-message", message);
    userInput.value = "";

    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_input: message }),
    })
        .then(response => response.json())
        .then(data => {
            addMessageToChat("bot-message", data.response);
        })
        .catch(error => {
            addMessageToChat("bot-message", "Sorry, something went wrong. Please try again.");
            console.error("Error:", error);
        });
}

function addMessageToChat(className, message) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.className = `message ${className}`;
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
