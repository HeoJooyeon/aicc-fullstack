//console.log("dom test");
document.getElementById("changeTextButton").addEventListener("click",function(){
    // document.getElementById("myParagraph").textContent = "the paragraph is changed";
       let paragraph1 = document.getElementById("myParagraph");
       paragraph1.textContent = "the paragraph is changed";
});

document.getElementById("highlightFirstCity").addEventListener("click",function(){
    // let firstChild1 = document.getElementById("citiesList").firstElementChild;
    // firstChild1.className = "highlight";
    // firstChild1.classList.add("highlight");
    document.getElementById("citiesList").classList.add("highlight");
});

document.getElementById("changeOrder").addEventListener("click",function(){
    document.getElementById("drink").classList.add("drinkColor");
    document.getElementById("drink").textContent = "Espresso";
});

document.getElementById("newShopping").addEventListener("click",function(){
    let newItem = document.createElement("li");
    newItem.textContent = "Eggs";   
    document.getElementById("shopList").appendChild(newItem);
});

document.getElementById("removeTask").addEventListener("click",function(){   
    // let lastItem = document.getElementById("taskList").lastElementChild;
    // document.getElementById("taskList").removeChild(lastItem);
    let taskList = document.getElementById("taskList");
    taskList.lastElementChild.remove();
});

document.getElementById("dblclick").addEventListener("dblclick",function(){   
   alert("hello1");
}); 

document.getElementById("teaList").addEventListener("click",function(event){        
    console.log("event.target::: "+event.target.textContent);
    //ul에 클릭이 되어 있으면 "event.target exists:"
    //클릭이 되어 잇지 않으면 "event.target is null or undefined"가 콘솔창에 뜨도록 코딩하세요
    if(event.target){
        console.log("event.target exists:",event.target);
    }else{
        console.log("event.target is null or undefined");
    }

    if(event.target && event.target.matches(".teaItem")){
        alert("you selected "+event.target.textContent);
    }
});

const teaItem = document.querySelectorAll(".teaItem");
for(let i=0;i<teaItem.length;i++){
    console.log(teaItem[i].textContent);
}
teaItem.forEach((item) => {
    console.log(item.textContent);
});

document.getElementById("fbsubmit").addEventListener("click",function(){
    let newFb = document.createElement("li");
    newFb.textContent = "Feedback is:" + document.getElementById("fbinput").value;
    document.getElementById("fbform").appendChild(newFb);
});

document.getElementById("feedbackForm").addEventListener("submit",function(event){
    event.preventDefault();
    let feedback = document.getElementById("feedbackInput").value;
    document.getElementById("feedbackDisplay").textContent = `Feedback is: ${feedback}`;
});

document.addEventListener("DOMContentLoaded", function(e){
    document.getElementById("domStatus").textContent = `DOM fully loaded`;
});

document.getElementById("togbtn").addEventListener("click",function(){
    let togtxt = document.getElementById("togtxt");
    // if(togtxt.className == "highlight"){
    //     togtxt.className = "";
    // }else{
    //     togtxt.className = "highlight";
    // }
    togtxt.classList.toggle("highlight");
});