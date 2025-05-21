//add => add
//node todo.js add "go to school"
//list => list
//remove => remove
//이외 => not found
const fs = require('fs');
const filePath = "./data/tasks.json";

const loadTasks = () => {
    try {
        const dataBuffer = fs.readFileSync(filePath);
        const dataJSON = dataBuffer.toString();
        return JSON.parse(dataJSON);
    } catch (error) {
        return [];
    }
};

const saveTasks = (tasks) => {
    const dataJSON1 = JSON.stringify(tasks);
    fs.writeFileSync(filePath, dataJSON1);
};

const addTask = (task) => {
    const tasks = loadTasks();
    tasks.push({task});
    saveTasks(tasks);
};

const listTask = () => {
    const listTasks = loadTasks();
    /*  for(let i=0;i<loadtasks.length;i++){
        console.log(`${i+1} - ${loadtasks[i].task}`);
        }; 
        let j = 0;
        while(j<loadtasks.length){
        console.log(`${j+1} - ${loadtasks[j].task}`);
        j++;
    }; */
    //task.forEach(element, index, array)
    listTasks.forEach((listTask, index) => {
        console.log(`${index+1} - ${listTask.task}`);
    });
};

const removeTask = (index) => {
    //node todo.js remove 3
    //1,2,3 이외의 번호를 remove 다음에 입력한 경우에는 "잘못된 번호 입력"이라는 말이 콘솔에 나오게 하시고, 1,2,3 중에 번호가 입력이 된 경우에는 위 remove함수가 실행되도록 하세요
    const removeTasks = loadTasks();
    if(index > 0 && index <= removeTasks.length){
        console.log(`remove is ${index} - ${removeTasks[index-1].task}`);
        removeTasks.splice(index-1, 1);
        saveTasks(removeTasks);
        listTask();
    }else{
        console.log(`remove fails because the input is not valid`);
    }
}


const command = process.argv[2];
const argument = process.argv[3];

if(command == 'add' || command == 'list' || command == 'remove'){
    if(command == 'add'){
        console.log(`command is ${command}`);
        addTask(argument);
    }else if(command == 'list'){
        console.log(`command is ${command}`);
        listTask();
    }else if(command == 'remove'){
        console.log(`command is ${command}`);
        removeTask(parseInt(argument));   
    }
}else{
    console.log(`command is not found`);
};

let fruits = ["Apple", "Banana", "Cherry", "Mango"];
fruits.splice(1,2); //인덱스 번호, 삭제할 갯수
console.log(`fruits:: ${fruits}`);
//fruits:: Apple,banana,Mango
fruits.splice(1,0,"banana");
console.log(`fruits:: ${fruits}`);
//위 데이터를 Apple, cherry, Mango로 바꿔 주세요
fruits.splice(1,1,"cherry");
console.log(`fruits:: ${fruits}`);