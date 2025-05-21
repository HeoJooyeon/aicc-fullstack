document.addEventListener("DOMContentLoaded", () => {
    const todoInput = document.getElementById("todo-input");
    const addTaskButton = document.getElementById("add-task-btn");
    const todoList = document.getElementById("todo-list");
    //parse
    //let taskcnt = 0;     //새로고침용 카운터
    let tasks = JSON.parse(localStorage.getItem("tasks"))||[];
    tasks.forEach(task => renderTask(task));
    taskcnt = 0;
    // for(let i=0;i<tasks.length;i++){
    //     renderTask(tasks[i]);
    // };

    addTaskButton.addEventListener("click", () => {
        const taskText = todoInput.value.trim();
        if(taskText === "") {return;}
        //else if(){}
        const newTask = {
            id: Date.now(), 
            text: taskText, 
            completed: false, //완료된
        };
        tasks.push(newTask);
        saveTasks();
        renderTask(newTask);
        todoInput.value = "";
        //console.log(tasks);
    });

    function saveTasks(){
        //console.log(typeof tasks);
        localStorage.setItem("tasks", JSON.stringify(tasks));
        //console.log(JSON.stringify(tasks));
        // let test1 = JSON.stringify(tasks);
        // console.log(typeof test1);
        //tasks 배열을 문자열로 변환해 localStorage에 저장하기 위해 JSON.stringify를 사용
    };

    function renderTask(task){
        // todoList.insertAdjacentHTML("beforeend",`<li>`+task.text+` <button>remove</button></li>`);
        const li = document.createElement("li");
        if(task.completed){
            li.classList.add("completed-style");
        };
        li.innerHTML = `<span>${task.text}</span><button>delete</button>`;
        todoList.appendChild(li);
        li.addEventListener("click",(e)=>{
            //if(e.target.matches("button")){return;};
            if(e.target.tagName === "BUTTON"){return;};
            task.completed = !task.completed;
            li.classList.toggle("completed-style");
            saveTasks();
        });

        li.querySelector("button").addEventListener("click", (e)=>{
            e.stopPropagation();
            tasks = tasks.filter((t) => t.id !== task.id);
            // console.log(task.id);
            li.remove();
            saveTasks();
        });  
        //새로고침
        // const child = todoList.children[taskcnt];
        // if(task.completed){
        //     child.className = "completed-style";
        // };
        // taskcnt++;
    };

    document.getElementById("clear-task-btn").addEventListener("click", (e)=>{
        e.stopPropagation();
        // localStorage.removeItem("tasks");
        // location.reload(true);
        localStorage.clear();
        todoList.innerHTML = "";
        console.log("All local storage data cleared!");
    });
});