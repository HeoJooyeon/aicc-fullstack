let tasks = [
    {id: 1, text: "Learn JavaScript"},
    {id: 2, text: "Practice React"},
    {id: 3, text: "Build a project"},
];
let task = {id: 2}; //task.id
tasks = tasks.filter((t) => t.id !== task.id);
console.log(tasks);

