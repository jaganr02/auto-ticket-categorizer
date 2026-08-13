const subjectInput = document.getElementById("subject");
const bodyInput = document.getElementById("body");

const classifyBtn = document.getElementById("classifyBtn");

const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");

const categoryElement = document.getElementById("category");
const confidenceElement = document.getElementById("confidence");
const priorityElement = document.getElementById("priority");
const statusElement = document.getElementById("status");

const confidenceText =
    document.getElementById("confidenceText");

const confidenceBar =
    document.getElementById("confidenceBar");

const statusBox =
    document.getElementById("statusBox");


classifyBtn.addEventListener("click", classifyTicket);


async function classifyTicket() {

    const subject = subjectInput.value.trim();
    const body = bodyInput.value.trim();


    if (!subject && !body) {

        alert(
            "Please enter a ticket subject or message."
        );

        return;
    }


    classifyBtn.disabled = true;
    classifyBtn.classList.add("loading");

    classifyBtn.querySelector("span:first-child").textContent =
        "Analyzing ticket...";


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                subject: subject,
                body: body
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Prediction failed."
            );

        }


        displayResult(data);


    } catch (error) {

        console.error(error);

        alert(
            "Unable to classify the ticket. Please try again."
        );

    } finally {

        classifyBtn.disabled = false;

        classifyBtn.classList.remove("loading");

        classifyBtn.querySelector("span:first-child").textContent =
            "Classify Ticket";

    }
}


function displayResult(data) {

    emptyState.classList.add("hidden");

    resultContent.classList.remove("hidden");


    // Category

    categoryElement.textContent =
        data.category.toUpperCase();


    // Confidence

    const confidence =
        Number(data.confidence);


    confidenceElement.textContent =
        `${confidence.toFixed(2)}%`;

    confidenceText.textContent =
        `${confidence.toFixed(2)}%`;


    confidenceBar.style.width =
        `${Math.min(confidence, 100)}%`;


    // Priority

    priorityElement.textContent =
        data.priority;


    // Status

    statusElement.textContent =
        data.status;


    if (data.status === "Needs human review") {

        statusBox.classList.add("review");

    } else {

        statusBox.classList.remove("review");

    }

}