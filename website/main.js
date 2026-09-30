(function () {
  var button = document.getElementById("copy-bibtex");
  var pre = document.getElementById("bibtex");
  if (!button || !pre) return;
  var original = button.textContent;
  button.addEventListener("click", function () {
    var text = pre.textContent;
    var done = function () {
      button.textContent = "Copied";
      button.classList.add("copied");
      setTimeout(function () {
        button.textContent = original;
        button.classList.remove("copied");
      }, 1800);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
    } else {
      fallback(text, done);
    }
  });
  function fallback(text, done) {
    var range = document.createRange();
    range.selectNodeContents(pre);
    var selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    try { document.execCommand("copy"); } catch (e) { /* leave text selected for manual copy */ }
    done();
  }
})();
