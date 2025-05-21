/* <ul style={{ backgroundColor: 'green', display: 'flex', listStyle: 'none', padding: 0, margin: 0}}>  */
import { CORE_CENCEPTS, TOPICS } from "./data.js";
import { CoreConcept } from "./components/CoreConcept.jsx";
import Header from "./components/Header.jsx";
import Examples from "./components/Examples.jsx";
import { useState } from "react";

function App() {
  //I'm making code
  const menuList = ['Component', 'JSX', 'Props', 'State'];
  const [currentList, setCurrentList] = useState(`${menuList[0]} is click `);
  const changeList = (item) => {
    //console.log(`${item} Click`);
    setCurrentList(`${item} is click ${item}${item}${item}${item}${item}${item}`);
  };

  const [ selectedTopic, setSelectedTopic] = useState('component');
  /* console.log(`selectedTopic:: ${selectedTopic}`); */
  return (
    <div>
      {/* <Header></Header> */}
      <Header/>
      <main>
        <section id="core-concepts">
          <h2>동적 UI 구성</h2>
          <ul>            
          {/* <CoreConcept 
              title = {CORE_CENCEPTS[0].title}
              description = {CORE_CENCEPTS[0].description}
              image ={CORE_CENCEPTS[0].image}
            /> */}
            <CoreConcept {...CORE_CENCEPTS[0]} />
            <CoreConcept {...CORE_CENCEPTS[1]} />
            <CoreConcept {...CORE_CENCEPTS[2]} />
            <CoreConcept {...CORE_CENCEPTS[3]} />
            <CoreConcept {...CORE_CENCEPTS[4]} />
          </ul>
        </section>
        <section id="example-concepts" style={{margin: '3rem auto'}}>
        <div className="hidden-div"></div>
          <h2 style={{textAlign: "left", height: '1vh', alignItems: 'center'}}>Examples</h2>
          <ul id="menu-concepts" style={{display: 'flex', listStyle: 'none', gap: '3rem', fontSize: '1rem', justifyContent: 'left', padding: 0, margin: '1rem 0'}}>
            {menuList.map((item, index) => (
              <li style={{padding: '0.5rem 1rem'}} key={index} onClick={()=>changeList(item)}>
                {item}
                </li>
              ))}
          </ul>
          <ul id="content-concepts" style={{ backgroundColor: 'violet', listStyle: 'none', height: '10vh'}}>
              <Examples description={currentList} />
          </ul>
        </section>
        <section id="examples">
          <h2>Examples</h2>
          <menu>
            <li><button onClick={() => setSelectedTopic('component')}>Component</button></li>
            <li><button onClick={() => setSelectedTopic('jsx')}>JSX</button></li>
           {/*  {console.log(`setSelectedTopic실행 후의 selectedTopic:: ${selectedTopic}`)} */}
            <li><button onClick={() => setSelectedTopic('props')}>Props</button></li>
            <li><button onClick={() => setSelectedTopic('state')}>State</button></li>
          </menu>
          <div id="tab-content">
            {/* <h3>{selectedTopic}</h3>
            <p>{topics[selectedTopic]}</p> */}
            <h3>{TOPICS[selectedTopic].title}</h3>
            <p>{TOPICS[selectedTopic].description}</p>
            <pre>
              <code>
                {TOPICS[selectedTopic].code}
              </code>
            </pre>
          </div>
          <div className="hidden-div"></div>
        </section>
      </main>
    </div>
  );
}

export default App;
