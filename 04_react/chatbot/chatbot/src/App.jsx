import React, { useState } from 'react'
import ChatBotApp from './components/ChatBotApp'
import ChatBotStart from './components/ChatBotStart'
import {v4 as uuidv4} from 'uuid'

const App = () => {
  const [ isChatting, setIsChatting ] = useState(false)
  const [chats, setChats] = useState([])
  const [activeChat, setActiveChat] = useState(null)

  const handleStartChat = () => {
    setIsChatting(true);
    if (!activeChat) { 
      createNewChat()
    }
  }

  const handleGoBack = () => {
    setIsChatting(false)
  }

  const createNewChat = (initialMessage = '') => {
      const newChat = {
        id: uuidv4(),
        displayId: `Chat ${new Date().toLocaleDateString('ko-KR')} ${new Date().toLocaleTimeString()}`,
        messages: initialMessage ? [{type: 'prompt', text: initialMessage, timestamp: new Date().toLocaleTimeString()}] : [],
      }
      setChats((prevChats) => [newChat, ...prevChats])
      setActiveChat(newChat.id)
  }
  
  return (
    <div className='container'>
      {isChatting ? 
       <ChatBotApp 
       onGoBack = {handleGoBack} 
       chats = {chats} 
       setChats = {setChats}
       activeChat = {activeChat}
       setActiveChat = {setActiveChat}
       onNewChat = {createNewChat}
       /> :
       <ChatBotStart onStartChat = {handleStartChat}/> }
    </div>
  )
}

export default App
