"""
Google Colab Keep-Alive Script
Prevents Colab from disconnecting during long training runs.

Usage:
1. Run this in a separate code cell before starting training
2. It will click the browser every 60 seconds to keep session alive
"""

from IPython.display import display, HTML

keep_alive_code = """
<script>
function ClickConnect(){
    console.log("Keeping Colab alive...");
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000); // Click every 60 seconds
console.log("Keep-alive started!");
</script>
<div>✅ Keep-alive script is running. Don't close this tab!</div>
"""

display(HTML(keep_alive_code))
print("Keep-alive script activated. Keep this tab open during training.")

