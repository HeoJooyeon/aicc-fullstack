document.getElementById("todolistForm").addEventListener("submit",function(event){
    event.preventDefault();
    let newTodo = document.createElement("li");
    newTodo.textContent = document.getElementById("todolistInput").value;
    let delbtn = document.createElement("button");
    delbtn.textContent = "delete";
    delbtn.className = "delbtn";
    newTodo.appendChild(delbtn);
    document.getElementById("todolistDisplay").appendChild(newTodo);
    const buttons = document.getElementsByClassName("delbtn");
    Array.from(buttons).forEach(button => {
        button.addEventListener("click", function() {
            this.className = "underline";
            this.parentElement.className = "underline";
        });
    });
});

