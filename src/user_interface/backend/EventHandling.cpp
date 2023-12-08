#include <boost/function.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/detail/atomic_count.hpp>
#include <iostream>

class EventQueue
{
public:
    typedef boost::intrusive_ptr<EventQueue> Ptr;
    typedef boost::function<void()> Event; // nullary functor
    struct Stopped {};

    EventQueue()
        : state_(STATE_READY)
        , ref_count_(0)
    {}

    void produce(Event event) {
        boost::mutex::scoped_lock lock(mtx_);
        assert(STATE_READY == state_);
        q_.push_back(event);
        cnd_.notify_one();
    }

    Event consume() {
        boost::mutex::scoped_lock lock(mtx_);
        while(STATE_READY == state_ && q_.empty())
            cnd_.wait(lock);
        if(!q_.empty()) {
            Event event(q_.front());
            q_.pop_front();
            return event;
        }
        // The queue has been stopped. Notify the waiting thread blocked in
        // EventQueue::stop(true) (if any) that the queue is empty now.
        cnd_.notify_all();
        throw Stopped();
    }

    void stop(bool wait_completion) {
        boost::mutex::scoped_lock lock(mtx_);
        state_ = STATE_STOPPED;
        cnd_.notify_all();
        if(wait_completion) {
            // Wait till all events have been consumed.
            while(!q_.empty())
                cnd_.wait(lock);
        }
        else {
            // Cancel all pending events.
            q_.clear();
        }
    }

private:
    // Disable construction on the stack. Because the event queue can be shared between multiple
    // producers and multiple consumers it must not be destroyed before the last reference to it
    // is released. This is best done through using a thread-safe smart pointer with shared
    // ownership semantics. Hence EventQueue must be allocated on the heap and held through
    // smart pointer EventQueue::Ptr.
    ~EventQueue() {
        this->stop(false);
    }

    friend void intrusive_ptr_add_ref(EventQueue* p) {
        ++p->ref_count_;
    }

    friend void intrusive_ptr_release(EventQueue* p) {
        if(!--p->ref_count_)
            delete p;
    }

    enum State {
        STATE_READY,
        STATE_STOPPED,
    };

    typedef std::list<Event> Queue;
    boost::mutex mtx_;
    boost::condition_variable cnd_;
    Queue q_;
    State state_;
    boost::detail::atomic_count ref_count_;
};


// ======= example of usage =======

void consumer_thread_function(EventQueue::Ptr event_queue)
try {
    for(;;) {
        EventQueue::Event event(event_queue->consume()); // get a new event 
        event(); // and invoke it
    }
}
catch(EventQueue::Stopped&) {
}

void some_work(int n) {
    std::cout << "thread " << boost::this_thread::get_id() << " : " << n << '\n';
    boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(500));
}

int main()
{
    some_work(1);

    // create an event queue that can be shared between multiple produces and multiple consumers
    EventQueue::Ptr queue(new EventQueue);

    // create two worker thread and pass them a pointer to queue
    boost::thread worker_thread_1(consumer_thread_function, queue);
    boost::thread worker_thread_2(consumer_thread_function, queue);

    // tell the worker threads to do something
    queue->produce(boost::bind(some_work, 2));
    queue->produce(boost::bind(some_work, 3));
    queue->produce(boost::bind(some_work, 4));

    // tell the queue to stop
    queue->stop(true);

    // wait till the workers thread stopped
    worker_thread_2.join();
    worker_thread_1.join();

    some_work(5);
}