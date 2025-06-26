document.addEventListener("DOMContentLoaded", function () {
    const targetWord = "QPyTorch";
    const replacementHTML = 'Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch';

    function replaceTextWithHTML(node) {
        let child = node.firstChild;
        while (child) {
            const next = child.nextSibling;

            if (child.nodeType === Node.TEXT_NODE) {
                if (child.nodeValue.includes(targetWord)) {
                    const parent = child.parentNode;
                    const temp = document.createElement("span");
                    // Escape any special chars and safely insert HTML
                    temp.innerHTML = child.nodeValue.replace(
                        new RegExp(`\\b${targetWord}\\b`, 'g'),
                        replacementHTML
                    );
                    while (temp.firstChild) {
                        parent.insertBefore(temp.firstChild, child);
                    }
                    parent.removeChild(child);
                }
            } else if (child.nodeType === Node.ELEMENT_NODE) {
                const tag = child.tagName.toLowerCase();
                if (!['script', 'style', 'pre', 'code'].includes(tag)) {
                    replaceTextWithHTML(child);
                }
            }
            child = next;
        }
    }

    replaceTextWithHTML(document.body);
});
